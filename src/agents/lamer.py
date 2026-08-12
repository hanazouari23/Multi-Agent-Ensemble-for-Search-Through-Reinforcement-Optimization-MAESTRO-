import os
import re
import time
import traceback
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..core.agents import AgentBase

load_dotenv()

API_KEY = os.getenv("LLMAPI_KEY")
BASE_URL = os.getenv("BASE_URL_HPC")
MODEL_NAME = os.getenv("MODEL_NAME_HPC_2")

if not API_KEY:
    raise RuntimeError("Missing environment variable: LLMAPI_KEY")
if not BASE_URL:
    raise RuntimeError("Missing environment variable: BASE_URL_HPC")
if not MODEL_NAME:
    raise RuntimeError("Missing environment variable: MODEL_NAME_HPC")


# ── LameR prompts (Table 1 of Shen et al., 2023) ─────────────────────────────
LAMER_SYSTEM_PROMPT = (
    "You are a helpful assistant for information retrieval. "
    "You will be given a query and a list of candidate answering passages, "
    "most of which are wrong. Write a single correct answering passage. "
    "Respond with exactly one concise sentence of plain text. "
)

LAMER_USER_TEMPLATE = (
    'Given a question "{query}" and its possible answering passages '
    "(most of these passages are wrong) enumerated as:\n"
    "{passages}\n\n"
    "please write a correct answering passage."
)

PASSAGE_CHARS = 512      # ~128-token truncation for candidate passages


def _clean_answer(text: str) -> str:
    """Strip Markdown and truncate to the first sentence, max 25 words."""
    if not text:
        return ""

    # Remove fenced code blocks entirely
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]*`", "", text)
    text = text.replace("`", "")

    # Remove all Markdown formatting characters (keep the words inside)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("*", "").replace("_", "")

    # Remove headers, bullets, numbering
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to FIRST SENTENCE only
    m = re.match(r"([^.!?]+[.!?])\s*", text)
    first_sentence = m.group(1).strip() if m else text

    # Hard word cap
    words = first_sentence.split()
    if len(words) > 25:
        first_sentence = " ".join(words[:25]) + "."

    return first_sentence


def _dedup_key(text: str) -> str:
    """Aggressive normalization: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LameRAgent(AgentBase):
    """
    LameR: "Large Language Models are Strong Zero-Shot Retriever" (Shen et al., 2023).

    Pipeline:
        1. Retrieve top-M passages with the original query via BM25.
        2. Prompt the LLM with the query + enumerated candidates; sample N answers.
        3. Clean and deduplicate the answers.
        4. Build augmented query q_bar = Concat(q, a1, q, a2, ..., q, aN)  (Eq. 5).
        5. Re-retrieve with q_bar via BM25.
    """

    def __init__(
        self,
        embed_model: SentenceTransformer,
        n_candidates: int = 5,       # N in Eq. (4); paper default = 5
        top_k_initial: int = 10,     # M in Eq. (3); paper default = 10
        top_k_final: int = 50,
        model_name: str = MODEL_NAME,
    ):
        super().__init__(agent_id=4, embed_model=embed_model)

        self.n_candidates = n_candidates
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self.model_name = model_name

        print(f"[LameR] LLM client config: base_url={BASE_URL}, model={self.model_name}")
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "MAESTRO-LameR",
                "X-Title": "LameR Agent",
            },
        )

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        original_query = query_features["query_text"]
        retriever = query_features["retriever"]
        top_k_final = query_features.get("top_k", self.top_k_final)

        # ── Step 1: initial BM25 retrieval for in-domain demonstrations (Eq. 3).
        init_doc_ids, _, init_corpus = retriever(original_query, self.top_k_initial)

        if not init_doc_ids:
            return {
                "new_query_text": original_query,
                "new_doc_ids": [],
                "new_doc_scores": np.array([], dtype=np.float32),
                "elapsed_time": time.time() - start_time,
                "cost": 0.0,
            }

        # Enumerate candidates exactly as in the paper prompt: "1.{c1} 2.{c2} ..."
        passages = []
        for rank, doc_id in enumerate(init_doc_ids[: self.top_k_initial], start=1):
            text = init_corpus.get(doc_id, "").strip()
            if text:
                passages.append(f"{rank}.{text[:PASSAGE_CHARS]}")
        passage_block = " ".join(passages)

        # ── Step 2: sample N independent answers from the LLM (Eq. 4).
        llm_start = time.time()
        candidates = self._generate_candidates(original_query, passage_block)
        llm_time = time.time() - llm_start

        # ── Step 3: augmented query, Eq. (5): q_bar = Concat(q, a1, q, a2, ...).
        if candidates:
            parts = []
            for answer in candidates:
                parts.append(original_query)
                parts.append(answer)
            augmented_query = " ".join(parts)
        else:
            augmented_query = original_query

        # ── Step 4: re-retrieve with the augmented query.
        retrieval_start = time.time()
        final_doc_ids, final_scores, _ = retriever(augmented_query, top_k_final)
        retrieval_time = time.time() - retrieval_start

        return {
            "new_query_text": augmented_query,
            "new_doc_ids": final_doc_ids,
            "new_doc_scores": np.array(final_scores, dtype=np.float32),
            "elapsed_time": llm_time + retrieval_time,
            "cost": float(len(candidates)),
        }

    def _generate_candidates(self, query: str, passage_block: str) -> List[str]:
        """
        Sample N independent answers in a single batched LLM call (Eq. 4).

        Uses the API's ``n`` parameter to get N independent completions
        (parallel decoding, 1 network round-trip). Falls back to an empty
        list if the endpoint does not support ``n > 1``.
        """
        system_msg = LAMER_SYSTEM_PROMPT
        user_msg = LAMER_USER_TEMPLATE.format(query=query, passages=passage_block)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=64,
                n=self.n_candidates,
            )
        except Exception as e:
            # Some HPC endpoints do not support n > 1.
            print(
                f"[LameR] Batched call failed ({type(e).__name__}). "
                f"Endpoint may not support n={self.n_candidates}."
            )
            traceback.print_exc()
            return []

        seen = set()
        candidates: List[str] = []

        for choice in response.choices:
            content = choice.message.content
            if not content or not content.strip():
                continue

            answer = _clean_answer(content)
            if len(answer) < 3:
                continue

            key = _dedup_key(answer)
            if key in seen:
                print(f"[LameR] Duplicate answer discarded: {answer[:60]!r}")
                continue

            seen.add(key)
            candidates.append(answer)

        print(f"[LameR] Final candidates ({len(candidates)}): {candidates}")
        return candidates
