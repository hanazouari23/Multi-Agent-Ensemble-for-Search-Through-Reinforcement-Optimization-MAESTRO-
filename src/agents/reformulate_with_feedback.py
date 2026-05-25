from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ..core.agents import AgentBase
import os
import numpy as np
import time
from typing import Dict, Any, List
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("LLMAPI_KEY")
BASE_URL = os.getenv("BASE_URL_HPC")  # or BASE_URL_UNI depending on the environment
MODEL_NAME = os.getenv("MODEL_NAME_HPC")  # or MODEL_NAME_UNI depending on the environment

if not API_KEY:
    raise RuntimeError("Missing environment variable: LLMAPI_KEY")
if not BASE_URL:
    raise RuntimeError("Missing environment variable: BASE_URL_HPC")
if not MODEL_NAME:
    raise RuntimeError("Missing environment variable: MODEL_NAME_HPC")


SYSTEM_PROMPT = """You are a query reformulation assistant for sparse lexical retrieval.

You receive:
1) An original user query.
2) The current top-k retrieved document snippets.

Your task is to rewrite the query into a single stronger search query that improves lexical overlap with relevant documents while preserving the original intent.

Instructions:
- Read the original query first and preserve its intent.
- Use the retrieved snippets only as feedback about how the topic may be expressed in the corpus.
- Keep all important entities, names, numbers, dates, locations, and quoted phrases from the original query unless they are clearly malformed.
- Rewrite the query into one concise keyword-focused search query.
- Make a meaningful improvement; do not merely reorder the original words.
- Add only 1 to 3 useful lexical anchors.
- Added anchors must be either:
  - explicitly supported by the snippets, or
  - obvious canonical equivalents of terms already present in the original query.
- Do not add unsupported facts.
- Do not add guessed phone numbers, counts, years, places, named entities, or domain assumptions unless they appear in the original query or directly in the snippets.
- If the snippets are noisy, weak, or off-topic, rewrite conservatively and stay close to the original query.
- Avoid query drift: do not let snippet details change the user’s intent.
- Do not write a sentence, explanation, list, JSON, labels, or multiple queries.
- Return exactly one rewritten query string and nothing else.
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
        Reformulate the query using top-k retrieved snippets as feedback.

        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'raw_results': Tuple - (doc_ids, doc_scores, corpus) from baseline retrieval
                - 'top_k': int - number of top documents to consider
                - 'retriever': callable - function to retrieve documents with the new query

        Returns:
            Dict with:
                - 'new_query_text': str - reformulated query
                - 'new_doc_ids': List[str] - document IDs retrieved with reformulated query
                - 'new_doc_scores': np.ndarray - new document scores
                - 'elapsed_time': float - time taken for reformulation + retrieval
        """
        original_query = query_features["query_text"]
        raw_results = query_features["raw_results"]
        top_k = query_features.get("top_k", 5)
        retriever = query_features["retriever"]

        if not raw_results or len(raw_results) < 3:
            raise RuntimeError(
                "raw_results must be a tuple of (doc_ids, doc_scores, corpus)"
            )

        doc_ids, doc_scores, corpus = raw_results

        start_time = time.time()
        reformulated_query = self._call_llm(
            query=original_query,
            doc_ids=doc_ids,
            corpus=corpus,
            top_k=top_k,
        )
        reformulation_time = time.time() - start_time

        retrieval_start = time.time()
        new_raw_results = retriever(reformulated_query, top_k)
        retrieval_time = time.time() - retrieval_start

        if not new_raw_results or len(new_raw_results) < 2:
            raise RuntimeError(
                "Retriever returned an unexpected format. Expected at least (doc_ids, scores)."
            )

        new_doc_ids = list(new_raw_results[0]) if new_raw_results[0] is not None else []
        new_doc_scores = (
            np.array(new_raw_results[1], dtype=np.float32)
            if new_raw_results[1] is not None
            else np.array([], dtype=np.float32)
        )

        total_elapsed = reformulation_time + retrieval_time

        return {
            "new_query_text": reformulated_query,
            "new_doc_ids": new_doc_ids,
            "new_doc_scores": new_doc_scores,
            "elapsed_time": total_elapsed,
        }

    def _call_llm(
        self,
        query: str,
        doc_ids: List[str],
        corpus: Dict[str, str],
        top_k: int,
    ) -> str:
        """
        Generate a rewritten query using the original query and top-k retrieved snippets.

        Args:
            query: Original query text
            doc_ids: List of retrieved document IDs
            corpus: Dict mapping doc_id -> document text
            top_k: Number of top documents to consider

        Returns:
            A single rewritten query string
        """
        doc_snippets = []

        for i, doc_id in enumerate(doc_ids[:top_k]):
            snippet_text = corpus.get(doc_id, "")
            if not snippet_text:
                continue

            snippet_text = " ".join(snippet_text.split())
            snippet_text = snippet_text[:300]
            doc_snippets.append(f"{i+1}. {snippet_text}")

        snippets_block = "\n".join(doc_snippets) if doc_snippets else "None"

        user_message = (
            f"Original query: {query}\n\n"
            f"Top-{len(doc_snippets)} retrieved snippets:\n"
            f"{snippets_block}\n\n"
            "Rewrite the query into one stronger sparse-retrieval query using useful lexical anchors from the snippets when appropriate."
        )

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=64,
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("No content returned from LLM")

        reformulated_query = content.strip().strip('"').strip()
        if not reformulated_query:
            raise RuntimeError("Empty reformulated query returned from LLM")

        return reformulated_query