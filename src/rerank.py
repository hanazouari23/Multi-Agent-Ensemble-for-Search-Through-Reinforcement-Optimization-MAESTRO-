from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from .agents import AgentBase
import numpy as np
import time

class RerankingAgent(AgentBase):
    """
    A re-ranking agent that uses a cross-encoder to re-score documents.
    """
    def __init__(self, agent_id: int, embed_model, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        super().__init__(agent_id, embed_model)
        self.model = CrossEncoder(model_name)

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, float]:
        query = query_features['query_text']
        documents = query_features.get('documents', [])
        
        if not documents:
            return {
                'delta_ndcg': 0.0,
                'delta_recall': 0.0,
                'delta_time': 0.0,
                'delta_cost': 0.0,
            }
        
        # Rerank the documents
        start_time = time.time()
        pairs = [[query, doc['text']] for doc in documents]
        scores = self.model.predict(pairs)
        elapsed_time = time.time() - start_time
        
        # Add scores
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by score descending
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Compute effects: average rerank score as proxy for improvement
        avg_score = np.mean(scores)
        
        return {
            'delta_ndcg': 0.1 * avg_score,  # Mock improvement based on score
            'delta_recall': 0.05 * avg_score,
            'delta_time': elapsed_time,
            'delta_cost': 0.1,  # Some cost for reranking
        }