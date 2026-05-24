from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ..core.agents import AgentBase
import os
import numpy as np
import time
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# system message sent to the LLM when reformulating queries
SYSTEM_PROMPT = (
    "You are an information retrieval expert. "
    "Given a user query and the top-k retrieved document snippets, generate 2-3 expansion terms "
    "that capture key concepts from the retrieved documents to better augment the original query. "
    "Return ONLY the expansion terms separated by spaces, no explanations."
)
API_KEY = os.getenv("LLMAPI_KEY")
BASE_URL = os.getenv("BASE_URL_HPC")  # or BASE_URL_UNI depending on the environment
MODEL_NAME = os.getenv("MODEL_NAME_HPC")  # or MODEL_NAME_UNI depending on the environment
class ReformulationAgent(AgentBase):
    def __init__(self, embed_model: SentenceTransformer):
        super().__init__(agent_id=0, embed_model=embed_model)
        load_dotenv()
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "MAESTRO-Query-Reformulator",
                "X-Title": "Query Reformulator",
            }
        )
    
    
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate query expansion terms from baseline retrieved documents.
        
        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'raw_results': Tuple - (doc_ids, doc_scores, corpus) from baseline retrieval
                
        Returns:
            Dict with:
                - 'new_query_text': str - original query + expansion terms
        """
        original_query = query_features['query_text']
        raw_results = query_features['raw_results']
        
        doc_ids, doc_scores, corpus = raw_results
        top_k = len(doc_ids)
        
        # Generate expansion terms from retrieved documents
        start_time = time.time()
        expansion_terms = self._call_llm(original_query, doc_ids, corpus, top_k)
        expansion_time = time.time() - start_time
        
        # Combine original query with expansion terms
        expanded_query = f"{original_query} {expansion_terms}"
        
        return {
            'new_query_text': expanded_query,
            'elapsed_time': expansion_time,
        }
    
    def _call_llm(self, query: str, doc_ids: List[str], corpus: Dict[str, str], top_k: int) -> str:
        """
        Generate expansion terms based on original query and top-k retrieved documents.
        
        Args:
            query: Original query text
            doc_ids: List of retrieved document IDs
            corpus: Dictionary mapping doc_ids to document text
            top_k: Number of top documents to consider
            
        Returns:
            Expansion terms as a single string
        """
        # Extract top-k document snippets
        doc_snippets = []
        for i, doc_id in enumerate(doc_ids[:top_k]):
            if doc_id in corpus:
                snippet = corpus[doc_id]
                # Limit snippet length for context window
                snippet = snippet[:300] if len(snippet) > 300 else snippet
                doc_snippets.append(f"Doc {i+1}: {snippet}")
        
        # Build user message with query and document context
        user_message = (
            f"Query: {query}\n\n"
            f"Top-{len(doc_snippets)} retrieved documents:\n"
            + "\n".join(doc_snippets)
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
        
        expansion_terms = content.strip()
        return expansion_terms
