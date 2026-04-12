from openai import OpenAI
from sentence_transformers import SentenceTransformer
from .agents import AgentBase
import os
import numpy as np
import time
from typing import Dict, Any, List, Tuple

# system message sent to the LLM when reformulating queries
SYSTEM_PROMPT = (
    "You are a query rewriting assistant. "
    "Given a user query, you rewrite it into a clearer, more complete search query. "
    "Preserve the original intent and return only the rewritten query."
)

class ReformulationAgent(AgentBase):
    def __init__(self, embed_model: SentenceTransformer):
        super().__init__(agent_id=0, embed_model=embed_model)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "your-notebook-url",
                "X-Title": "Query Reformulator",
            }
        )
    
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reformulate the query and retrieve new documents.
        
        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'retriever': callable - function to retrieve documents
                
        Returns:
            Dict with:
                - 'new_query_text': str - reformulated query
                - 'new_doc_ids': List[str] - new document IDs
                - 'new_doc_scores': np.ndarray - new document scores
                - 'elapsed_time': float - time taken for reformulation + retrieval
        """
        original_query = query_features['query_text']
        retriever = query_features['retriever']
        
        # 1. Reformulate query with LLM
        start_time = time.time()
        reformulated_query = self._call_llm(original_query)
        reformulation_time = time.time() - start_time
        
        # 2. Retrieve new documents with reformulated query
        retrieval_start = time.time()
        raw_results = retriever(reformulated_query)
        retrieval_time = time.time() - retrieval_start
        
        # 3. Extract doc_ids and scores
        new_doc_ids = [doc_id for doc_id, _ in raw_results]
        new_doc_scores = np.array([score for _, score in raw_results], dtype=np.float32)
        
        total_elapsed = reformulation_time + retrieval_time
        
        return {
            'new_query_text': reformulated_query,
            'new_doc_ids': new_doc_ids,
            'new_doc_scores': new_doc_scores,
            'elapsed_time': total_elapsed,
        }
    
    def _call_llm(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek/deepseek-v3.2",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        content = response.choices[0].message.content.strip() 
        if not content:
            raise RuntimeError("No choices returned")  
        # Get token counts from usage (this DOES work with OpenRouter)
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
    
        # For OpenRouter: query their pricing endpoint or use local cache
        # Option 1: Use cached pricing (requires initial fetch)
        # Option 2: Hit /billing/usage endpoint after the fact
        # estimated_cost = self._estimate_cost(prompt_tokens, completion_tokens)

        # return content, estimated_cost
        return content
    
    # def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
    #     """Get pricing from OpenRouter or use cached values"""
    #     # Check OpenRouter's pricing API or store locally
    #     # Example: deepseek-v3.2 might be $0.27/$0.81 per 1M tokens (example rates)
    #     DEEPSEEK_PRICING = {"prompt": 0.247, "completion": 0.472}  # $ per 1M tokens

    #     cost = (prompt_tokens * DEEPSEEK_PRICING["prompt"] + 
    #             completion_tokens * DEEPSEEK_PRICING["completion"]) / 1_000_000
    #     return cost


