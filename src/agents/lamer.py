import time
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..core.agents import AgentBase


MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-FP8"

# ── LameR Prompts ─────────────────────────────────────────────────────────────
# The LLM is NOT asked to rewrite the query. It is shown the query plus the
# top-k BM25 passages as in-domain demonstrations and asked to generate
# plausible answers. Those answers are concatenated with the original query
# and submitted back to BM25.
LAMER_SYSTEM_PROMPT = (
    "You are a helpful reading assistant. "
    "Given a question and a set of passages retrieved by a search engine, "
    "generate up to {n_candidates} concise, plausible answers to the question. "
    "Each answer should be a short phrase or sentence that directly answers the question. "
    "Return exactly one answer per line. Do not enumerate them. "
    "If the passages do not contain enough information, generate likely answers based on the question alone."
)

LAMER_USER_TEMPLATE = (
    "Question: {query}\n\n"
    "Retrieved passages:\n{passages}\n\n"
    "Generate up to {n_candidates} plausible answers (one per line):"
)


class LameRAgent(AgentBase):
    """
    LameR: "LLMs are Strong Zero-Shot Retrievers" (Shen et al., 2023).

    Pipeline:
        1. Retrieve top-k passages with the original query via BM25.
        2. Prompt an LLM with the query and those passages as demonstrations.
        3. Parse the generated candidate answers.
        4. Concatenate the original query with the candidate answers.
        5. Re-retrieve with the augmented composite query via BM25.

    The agent follows the MAESTRO ``AgentBase`` contract and can be evaluated
    in isolation before being wired into the RL ensemble.
    """

    def __init__(
        self,
        embed_model: SentenceTransformer,
        n_candidates: int = 3,
        top_k_initial: int = 10,
        top_k_final: int = 50,
    ):
        # agent_id=4 reserves a slot outside the current {QR, RR, PRF, STOP} set.
        super().__init__(agent_id=4, embed_model=embed_model)

        self.n_candidates = n_candidates
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LameR zero-shot retrieval pipeline.

        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query.
                - 'retriever': callable - ``(query, top_k) -> (doc_ids, scores, corpus_data)``.
                - 'top_k': int (optional) - final retrieval window; overrides ``top_k_final``.

        Returns:
            Dict with:
                - 'new_query_text': str - augmented composite query.
                - 'new_doc_ids': List[str] - documents retrieved with the augmented query.
                - 'new_doc_scores': np.ndarray - BM25 scores for the final ranking.
                - 'elapsed_time': float - wall-clock seconds for the full pipeline.
                - 'cost': float - fixed LLM-call cost placeholder.
        """
        start_time = time.time()

        original_query = query_features["query_text"]
        retriever = query_features["retriever"]
        top_k_final = query_features.get("top_k", self.top_k_final)

        # ── Step 1: Initial BM25 retrieval to collect in-domain demonstrations.
        init_doc_ids, _, init_corpus = retriever(original_query, self.top_k_initial)

        if not init_doc_ids:
            return {
                "new_query_text": original_query,
                "new_doc_ids": [],
                "new_doc_scores": np.array([], dtype=np.float32),
                "elapsed_time": time.time() - start_time,
                "cost": 0.0,
            }

        passages = []
        for rank, doc_id in enumerate(init_doc_ids[: self.top_k_initial], start=1):
            text = init_corpus.get(doc_id, "").strip()
            if text:
                # Truncate to keep the prompt compact and cheap.
                passages.append(f"[{rank}] {text[:300]}")
        passage_block = "\n".join(passages)

        # ── Step 2: LLM generates candidate answers from query + passages.
        llm_start = time.time()
        candidates = self._generate_candidates(original_query, passage_block)
        llm_time = time.time() - llm_start

        # ── Step 3: Build the augmented composite query.
        if candidates:
            augmented_query = original_query + " " + " ".join(candidates)
        else:
            augmented_query = original_query

        # ── Step 4: Re-retrieve with the augmented query.
        retrieval_start = time.time()
        final_doc_ids, final_scores, _ = retriever(augmented_query, top_k_final)
        retrieval_time = time.time() - retrieval_start

        total_elapsed = llm_time + retrieval_time

        return {
            "new_query_text": augmented_query,
            "new_doc_ids": final_doc_ids,
            "new_doc_scores": np.array(final_scores, dtype=np.float32),
            "elapsed_time": total_elapsed,
            "cost": 1.0,  # Placeholder; calibrate against actual token spend.
        }

    def _generate_candidates(self, query: str, passage_block: str) -> List[str]:
        """
        Call the local Llama model and parse one candidate answer per line.

        Args:
            query: Original query text.
            passage_block: Formatted top-k passages from the initial retrieval.

        Returns:
            A deduplicated list of up to ``n_candidates`` answer strings.
        """
        system_msg = LAMER_SYSTEM_PROMPT.format(n_candidates=self.n_candidates)
        user_msg = LAMER_USER_TEMPLATE.format(
            query=query,
            passages=passage_block,
            n_candidates=self.n_candidates,
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            content = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
            if not content:
                return []
        except Exception:
            # Gracefully degrade to the original query if the LLM call fails.
            return []

        candidates = []
        for line in content.strip().splitlines():
            line = line.strip()
            # Strip common enumeration markers, e.g. "1.", "-", "*", "[1]".
            line = line.lstrip("0123456789.-*)[] ").strip()
            if line and len(line) > 2:
                candidates.append(line)

        # Deduplicate (case-insensitive) while preserving order.
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
                if len(unique_candidates) >= self.n_candidates:
                    break

        return unique_candidates
