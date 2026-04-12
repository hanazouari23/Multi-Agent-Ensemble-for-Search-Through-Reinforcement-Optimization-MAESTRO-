"""
Click-Prior Reweighting Agent

Reweights search results by boosting scores of documents that received
clicks in the ORCAS dataset, pushing the most-clicked documents higher
in the ranking.
"""

from typing import Dict, Any, List
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from ..core.agents import AgentBase


class ClickPriorAgent(AgentBase):
    """
    Agent that reweights retrieval results using ORCAS click priors.
    
    For each document, adds a click-based prior score to the original BM25
    score, then re-ranks. This empirically improves user satisfaction by
    promoting clicked documents.
    
    Parameters
    ----------
    embed_model : SentenceTransformer
        Shared embedding model (from AgentBase interface)
    beta : float
        Weight for prior boost: new_score = old_score + β * prior
        Range: typically 0.01 to 1.0. Default 0.1 adds modest boost.
    """
    
    def __init__(
        self,
        embed_model: SentenceTransformer,
        beta: float = 0.1
    ):
        super().__init__(agent_id=2, embed_model=embed_model)  # agent_id=2 for CP
        self.beta = beta
    
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply click-prior reweighting to ranked documents.
        
        This agent boosts document scores based on ORCAS click priors,
        then re-ranks. Returns the reweighted results.
        
        Parameters
        ----------
        query_features : Dict[str, Any]
            Must contain:
            - 'query_text': str – the search query
            - 'doc_ids': List[str] – current ranked doc IDs
            - 'doc_scores': np.ndarray – current BM25/retrieval scores
            - 'orcas_index': Dict[str, List[str]] – ORCAS click data
        
        Returns
        -------
        Dict[str, Any]
            - 'new_doc_ids': reranked docs after click-prior boost
            - 'new_doc_scores': reweighted scores
            - 'prior_scores': click-prior values used (for logging)
            - 'elapsed_time': time taken (seconds)
        """
        start_time = time.time()
        
        query_text = query_features.get('query_text', '')
        doc_ids = query_features.get('doc_ids', [])
        doc_scores = query_features.get('doc_scores', np.array([], dtype=np.float32))
        orcas_index = query_features.get('orcas_index', {})
        
        # Handle empty inputs
        if len(doc_ids) == 0 or doc_scores.size == 0:
            return {
                'new_doc_ids': doc_ids,
                'new_doc_scores': doc_scores,
                'prior_scores': np.array([], dtype=np.float32),
                'elapsed_time': 0.0,
            }
        
        # Get prior scores from ORCAS index
        prior_scores = self._get_prior_scores(query_text, doc_ids, orcas_index)
        
        # Boost and rerank
        doc_scores_array = np.asarray(doc_scores, dtype=np.float32)
        boosted_scores = doc_scores_array + self.beta * prior_scores
        
        # Sort by boosted scores (descending)
        order = np.argsort(boosted_scores)[::-1]
        reranked_doc_ids = [doc_ids[i] for i in order]
        reranked_scores = boosted_scores[order]
        
        elapsed_time = time.time() - start_time
        
        return {
            'new_doc_ids': reranked_doc_ids,
            'new_doc_scores': reranked_scores,
            'prior_scores': prior_scores,
            'elapsed_time': elapsed_time,
        }
    
    def _get_prior_scores(self, query: str, doc_ids: List[str], orcas_index: Dict[str, List[str]]) -> np.ndarray:
        """
        Return binary prior scores for each document.
        
        Parameters
        ----------
        query : str
            Query string (may be case-sensitive)
        doc_ids : List[str]
            Document IDs to score
        orcas_index : Dict[str, List[str]]
            ORCAS click data: query → [clicked_doc_ids]
        
        Returns
        -------
        np.ndarray of shape (len(doc_ids),), dtype float32
            1.0 if doc_id was clicked for query in ORCAS, else 0.0
        """
        # Attempt exact match; fall back to normalized query
        clicked_docs = orcas_index.get(query, None)
        if clicked_docs is None:
            # Try normalized (lowercase, stripped)
            normalized_query = query.lower().strip()
            clicked_docs = orcas_index.get(normalized_query, [])
        
        clicked_set = set(clicked_docs)
        return np.array(
            [1.0 if doc_id in clicked_set else 0.0 for doc_id in doc_ids],
            dtype=np.float32
        )
