import os
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


# ── LameR Prompts ─────────────────────────────────────────────────────────────
# The LLM is NOT asked to rewrite the query. It is shown the query plus the
# top-k BM25 passages as in-domain demonstrations and asked to generate
# plausible answers. Those answers are concatenated with the original query
# and submitted back to BM25.
LAMER_SYSTEM_PROMPT = (
    "You are a helpful assistant for information retrieval. "
    "You will be given a question and a list of candidate answering passages, "
    "most of which are wrong. Write a single correct answering passage."
)

LAMER_USER_TEMPLATE = (
    'Give a question "{query}" and its possible answering passages '
    "(most of these passages are wrong) enumerated as:\n"
    "{passages}\n\n"
    "please write a correct answering passage."
)


class LameRAgent(AgentBase):
    def __init__(
        self,
        embed_model: SentenceTransformer,
        n_candidates: int = 5,          # N in Eq.(4), paper default = 5
        top_k_initial: int = 10,        # M in Eq.(3), paper default = 10
        top_k_final: int = 50,
        model_name: str = MODEL_NAME,
    ):
        super().__init__(agent_id=4, embed_model=embed_model)
        self.n_candidates = n_candidates
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self.model_name = model_name
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={"HTTP-Referer": "MAESTRO-LameR", "X-Title": "LameR Agent"},
        )

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        original_query = query_features["query_text"]
        retriever = query_features["retriever"]
        top_k_final = query_features.get("top_k", self.top_k_final)

        init_doc_ids, _, init_corpus = retriever(original_query, self.top_k_initial)

        if not init_doc_ids:
            return {
                "new_query_text": original_query,
                "new_doc_ids": [],
                "new_doc_scores": np.array([], dtype=np.float32),
                "elapsed_time": time.time() - start_time,
                "cost": 0.0,
            }

        # Table 1 style enumeration: "1.{c1} 2.{c2} ..."
        passages = []
        for rank, doc_id in enumerate(init_doc_ids[: self.top_k_initial], start=1):
            text = init_corpus.get(doc_id, "").strip()
            if text:
                passages.append(f"{rank}.{text[:512]}")  # 128-token-ish truncation
        passage_block = " ".join(passages)

        llm_start = time.time()
        candidates = self._generate_candidates(original_query, passage_block)
        llm_time = time.time() - llm_start

        # Eq.(5): q̄ = Concat(q, a1, q, a2, ..., q, aN)
        if candidates:
            parts = []
            for a in candidates:
                parts.append(original_query)
                parts.append(a)
            augmented_query = " ".join(parts)
        else:
            augmented_query = original_query

        retrieval_start = time.time()
        final_doc_ids, final_scores, _ = retriever(augmented_query, top_k_final)
        retrieval_time = time.time() - retrieval_start

        return {
            "new_query_text": augmented_query,
            "new_doc_ids": final_doc_ids,
            "new_doc_scores": np.array(final_scores, dtype=np.float32),
            "elapsed_time": llm_time + retrieval_time,
            "cost": float(self.n_candidates),
        }

    def _generate_candidates(self, query: str, passage_block: str) -> List[str]:
        """Sample N independent answers (Eq. 4: a ~ LLM(p(t,q,C^q)), N times)."""
        system_msg = LAMER_SYSTEM_PROMPT
        user_msg = LAMER_USER_TEMPLATE.format(query=query, passages=passage_block)

        candidates: List[str] = []
        for _ in range(self.n_candidates):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    candidates.append(content.strip())
            except Exception as e:
                traceback.print_exc()
                continue

        return candidates