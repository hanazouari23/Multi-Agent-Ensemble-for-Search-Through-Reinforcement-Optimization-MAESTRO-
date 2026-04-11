from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sympy import content
from .agents import AgentBase
import os
import numpy as np
import time
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
    
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, float]:
        original_query = query_features['query_text']  # add this to features
        orig_embedding = query_features['embedding']
        
        # 1. Call LLM to reformulate (with timing)
        start_time = time.time()
        rewritten_query = self._call_llm(original_query)
        elapsed_time = time.time() - start_time
        
        # 2. Re-embed the NEW query (critical for state update!)
        new_embedding = self.embed_model.encode(rewritten_query)
        
        # 3. Simulate effects (replace with real eval)
        # For now: mock based on embedding similarity + priors
        sim = np.dot(orig_embedding[:512], new_embedding[:512]) / (
            np.linalg.norm(orig_embedding[:512]) * np.linalg.norm(new_embedding[:512])
        )  # cosine sim first 512 dims
        
        return {

            'delta_ndcg': 0.05 * sim,      # Better query → better NDCG
            'delta_recall': 0.02 * sim,
            'delta_time': elapsed_time,    #LLM latency (seconds)
            'delta_cost': 0.8,        #API cost 
            'satisfaction': query_features['prior_cov'] * sim,
            'new_query_text': rewritten_query,    # Pass back for logging
            'new_embedding': new_embedding,       # Update query_features!
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


