from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ..core.agents import AgentBase
import os
import numpy as np
import time
from typing import Dict, Any
from dotenv import load_dotenv


load_dotenv()


# Environment
API_KEY = os.getenv("LLMAPI_KEY")
BASE_URL = os.getenv("BASE_URL_HPC")  # or BASE_URL_UNI depending on the environment
MODEL_NAME = os.getenv("MODEL_NAME_HPC")  # or MODEL_NAME_UNI depending on the environment


if not API_KEY:
    raise RuntimeError("Missing environment variable: LLMAPI_KEY")
if not BASE_URL:
    raise RuntimeError("Missing environment variable: BASE_URL_HPC")
if not MODEL_NAME:
    raise RuntimeError("Missing environment variable: MODEL_NAME_HPC")


# System message sent to the LLM when generating expansion terms
SYSTEM_PROMPT = """You are a query expansion assistant for sparse lexical retrieval.

The input query currently fails to retrieve relevant results.
Generate a small set of expansion terms that can be appended to the original query to improve lexical overlap with relevant documents.

Rules:
- Preserve the original intent.
- Keep all entities, names, numbers, dates, locations, and quoted phrases from the original query unchanged.
- Generate only 2 to 4 short expansion terms or keyphrases.
- Prefer concise keyword-style expansions.
- Correct spelling or malformed wording only if needed for the expansion terms.
- Add highly specific terms that are strongly likely to appear in relevant documents.
- Do not add unsupported facts, guessed locations, guessed dates, guessed entities, phone numbers, exact counts, or brand names unless they are already present in the original query.
- Do not repeat the original query.
- Do not return a rewritten full query.
- Do not return explanations, lists, JSON, labels, punctuation-heavy formatting, or sentence-like text.
- Return only the expansion terms, separated by spaces.
"""


class ReformulationAgent(AgentBase):
    def __init__(self, embed_model: SentenceTransformer):
        super().__init__(agent_id=0, embed_model=embed_model)

        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "MAESTRO-Query-Reformulator",
                "X-Title": "Query Reformulator",
            },
        )

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate expansion terms from the original query and retrieve new documents.

        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'retriever': callable - function to retrieve documents
                - 'top_k': int (optional) - number of top documents to retrieve

        Returns:
            Dict with:
                - 'new_query_text': str - original query plus expansion terms
                - 'new_doc_ids': list[str] - new document IDs
                - 'new_doc_scores': np.ndarray - new document scores
                - 'elapsed_time': float - time taken for expansion + retrieval
        """
        original_query = query_features["query_text"]
        retriever = query_features["retriever"]
        top_k = query_features.get("top_k", 5)

        start_time = time.time()
        expansion_terms = self._call_llm(original_query)
        expansion_time = time.time() - start_time

        expanded_query = f"{original_query} {expansion_terms}".strip()

        retrieval_start = time.time()
        raw_results = retriever(expanded_query, top_k)
        retrieval_time = time.time() - retrieval_start

        if not raw_results or len(raw_results) < 2:
            raise RuntimeError(
                "Retriever returned an unexpected format. Expected at least (doc_ids, scores)."
            )

        doc_ids = raw_results[0]
        scores = raw_results[1]

        new_doc_ids = list(doc_ids) if doc_ids is not None else []
        new_doc_scores = (
            np.array(scores, dtype=np.float32)
            if scores is not None
            else np.array([], dtype=np.float32)
        )

        total_elapsed = expansion_time + retrieval_time

        return {
            "new_query_text": expanded_query,
            "new_doc_ids": new_doc_ids,
            "new_doc_scores": new_doc_scores,
            "elapsed_time": total_elapsed,
        }

    def _call_llm(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Generate expansion terms for this query:\n{query}",
                },
            ],
            temperature=0.2,
            max_tokens=32,
        )

        message = response.choices[0].message.content
        if message is None:
            raise RuntimeError("No content returned from LLM")

        content = " ".join(message.strip().strip('"').split())
        if not content:
            raise RuntimeError("Empty expansion terms returned from LLM")

        return content