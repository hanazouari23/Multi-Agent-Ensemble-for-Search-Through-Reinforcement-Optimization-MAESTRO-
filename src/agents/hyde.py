import os
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..core.agents import AgentBase


# API configuration is read from the environment (same pattern as LameRAgent).
load_dotenv()

API_KEY = os.getenv("LLMAPI_KEY")
BASE_URL = os.getenv("BASE_URL_HPC")
MODEL_NAME = os.getenv("MODEL_NAME_HPC_2")

if not API_KEY:
    raise RuntimeError("Missing environment variable: LLMAPI_KEY")
if not BASE_URL:
    raise RuntimeError("Missing environment variable: BASE_URL_HPC")
if not MODEL_NAME:
    raise RuntimeError("Missing environment variable: MODEL_NAME_HPC_3")

# ── HyDE Prompts ──────────────────────────────────────────────────────────────
# The LLM is asked to hallucinate a short passage that would answer the
# question, using terminology likely to appear in relevant documents.
HYDE_SYSTEM_PROMPT = "You are a helpful technical writing assistant."

HYDE_USER_TEMPLATE = (
    "Write a short hypothetical passage that would answer this question.\n"
    "Use terminology likely to occur in relevant documents.\n"
    "Do not mention that the passage is hypothetical.\n\n"
    "Question: {query}\n\n"
    "Generated pseudo-document:"
)


def _reciprocal_rank_fusion(
    result_lists: List[List[str]],
    top_k: int,
    rrf_k: int = 60,
) -> Tuple[List[str], np.ndarray]:
    """
    Fuse ranked document-ID lists using Reciprocal Rank Fusion.

    Adapted from ``PRFAgent._reciprocal_rank_fusion`` in ``prf.py``.
    A document receives 1 / (rrf_k + rank) for every list in which it occurs.
    Ranks start at 1.
    """
    fused_scores = defaultdict(float)

    for doc_ids in result_lists:
        for rank, doc_id in enumerate(doc_ids, start=1):
            fused_scores[doc_id] += 1.0 / (rrf_k + rank)

    ranked_docs = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),  # deterministic tie-break
    )[:top_k]

    doc_ids = [doc_id for doc_id, _ in ranked_docs]
    scores = np.asarray(
        [score for _, score in ranked_docs],
        dtype=np.float32,
    )

    return doc_ids, scores


class HyDEAgent(AgentBase):
    """
    HyDE: Hypothetical Document Embeddings (Lewis et al., 2023), re-implemented
    for a sparse BM25 retriever.

    Pipeline:
        1. Ask an LLM API to write a short hypothetical passage that answers the
           query, using terminology likely to occur in technical documents.
        2. Retrieve top-k documents with the original query.
        3. Retrieve top-k documents with the generated pseudo-document.
        4. Fuse the two rankings with reciprocal rank fusion (RRF).

    The agent follows the MAESTRO ``AgentBase`` contract and can be evaluated
    in isolation before being wired into the RL ensemble.
    """

    def __init__(
        self,
        embed_model: SentenceTransformer,
        top_k: int = 50,
        rrf_k: int = 60,
    ):
        # agent_id=5 keeps the agent outside the current {QR, RR, PRF, STOP} set.
        super().__init__(agent_id=5, embed_model=embed_model)

        self.top_k = top_k
        self.rrf_k = rrf_k

        print(f"[HyDE] LLM client config: base_url={BASE_URL}, model={MODEL_NAME}")
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "MAESTRO-HyDE",
                "X-Title": "HyDE Pseudo-Document Generator",
            },
        )

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the HyDE pseudo-document retrieval pipeline.

        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query.
                - 'retriever': callable - ``(query, top_k) -> (doc_ids, scores, corpus_data)``.
                - 'top_k': int (optional) - retrieval/fusion window; overrides ``top_k``.

        Returns:
            Dict with:
                - 'new_query_text': str - the generated pseudo-document.
                - 'new_doc_ids': List[str] - fused document IDs.
                - 'new_doc_scores': np.ndarray - RRF scores for the fused ranking.
                - 'elapsed_time': float - wall-clock seconds for the full pipeline.
                - 'cost': float - fixed LLM-call cost placeholder.
        """
        start_time = time.time()

        original_query = query_features["query_text"]
        retriever = query_features["retriever"]
        top_k = query_features.get("top_k", self.top_k)

        # ── Step 1: Retrieve with the original query.
        query_doc_ids, query_scores, _ = retriever(original_query, top_k)

        if not query_doc_ids:
            return {
                "new_query_text": original_query,
                "new_doc_ids": [],
                "new_doc_scores": np.array([], dtype=np.float32),
                "elapsed_time": time.time() - start_time,
                "cost": 0.0,
            }

        # ── Step 2: Generate a hypothetical passage with the LLM API.
        llm_start = time.time()
        pseudo_document = self._generate_passage(original_query)
        llm_time = time.time() - llm_start

        # ── Step 3: Retrieve with the pseudo-document.
        retrieval_start = time.time()
        if pseudo_document:
            passage_doc_ids, _, _ = retriever(pseudo_document, top_k)
        else:
            passage_doc_ids = []
        retrieval_time = time.time() - retrieval_start

        # ── Step 4: Fuse the two rankings with RRF.
        if passage_doc_ids:
            fused_doc_ids, fused_scores = _reciprocal_rank_fusion(
                result_lists=[query_doc_ids, passage_doc_ids],
                top_k=top_k,
                rrf_k=self.rrf_k,
            )
        else:
            fused_doc_ids = query_doc_ids
            fused_scores = np.asarray(query_scores, dtype=np.float32)

        total_elapsed = llm_time + retrieval_time

        return {
            "new_query_text": pseudo_document or original_query,
            "new_doc_ids": fused_doc_ids,
            "new_doc_scores": fused_scores,
            "elapsed_time": total_elapsed,
            "cost": 1.0,  # Placeholder; calibrate against actual token spend.
        }

    def _generate_passage(self, query: str) -> str:
        """
        Call the LLM API and return a single hypothetical passage.

        Args:
            query: Original query text.

        Returns:
            The generated pseudo-document string, or an empty string on failure.
        """
        messages = [
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": HYDE_USER_TEMPLATE.format(query=query)},
        ]

        try:
            print(f"[HyDE] Calling LLM API with model={MODEL_NAME}")
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
            )
            print(f"[HyDE] LLM API response received: {type(response)}")
            content = response.choices[0].message.content
            if content is None:
                print("[HyDE] LLM response content is None")
                return ""
            content = content.strip()
            if not content:
                print("[HyDE] LLM response content is empty after strip")
                return ""
        except Exception as e:
            print(f"[HyDE] API call failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            return ""

        # Keep only the first paragraph if the model emits extra commentary.
        first_block = content.split("\n\n")[0].strip()
        return first_block
