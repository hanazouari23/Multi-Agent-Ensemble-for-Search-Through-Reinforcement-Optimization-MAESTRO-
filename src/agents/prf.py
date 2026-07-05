
import time
from typing import Any, Dict, List

import numpy as np

from src.core.agents import AgentBase
from sklearn.feature_extraction.text import TfidfVectorizer


class PRFAgent(AgentBase):

    """Pseudo-Relevance Feedback (PRF) agent: Takes the top-k BM25 documents and extracts the most 
    discriminative terms from their content using TF-IDF weighting. These terms are used to compute 
    PRF-based scores for documents, which are then re-ranked to surface documents the initial search missed. """

    def __init__(self, embed_model, num_expansion_terms):
        super().__init__(agent_id=2, embed_model=embed_model)
        self.num_expansion_terms = num_expansion_terms
    
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply PRF: extract expansion terms from top-k documents, expand query, re-retrieve.
        
        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'retriever': callable - function to retrieve documents
                - 'top_k_rerank': int - TOP_K: number of initial documents to retrieve
                - 'top_k_prf': int - TOP_K_PRF: number of top documents to extract expansion terms from
                
        Returns:
            Dict with:
                - 'new_query_text': str - expanded query
                - 'new_doc_ids': List[str] - document IDs from re-retrieval
                - 'new_doc_scores': np.ndarray - scores from re-retrieval
                - 'elapsed_time': float - time taken for PRF process
        """
        start_time = time.time()
        
        retriever = query_features['retriever']
        original_query = query_features['query_text']
        top_k = query_features['top_k_rerank']
        top_k_prf = query_features['top_k_prf']
        
        # 1. Initial retrieval: get TOP_K documents
        initial_results = retriever(original_query, top_k)
        
        if not initial_results or len(initial_results[0]) == 0:
            elapsed_time = time.time() - start_time
            return {
                'new_query_text': original_query,
                'new_doc_ids': [],
                'new_doc_scores': np.array([], dtype=np.float32),
                'elapsed_time': elapsed_time,
            }
        
        initial_doc_ids = initial_results[0]
        initial_scores = initial_results[1]
        segments_dict = initial_results[2] if len(initial_results) > 2 else {}
        
        # 2. Extract TOP_K_PRF documents from initial TOP_K results
        prf_doc_ids = initial_doc_ids[:min(top_k_prf, len(initial_doc_ids))]
        prf_segments = [segments_dict.get(doc_id, "") for doc_id in prf_doc_ids]
        
        # 3. Extract NUM_EXPANSION_TERMS expansion terms using TF-IDF
        expansion_terms = self._extract_expansion_terms(original_query, prf_segments)
        
        # 4. Expand query with extracted terms
        new_query_text = original_query + " " + " ".join(expansion_terms) if expansion_terms else original_query
        
        # 5. Re-retrieve TOP_K documents using expanded query
        new_results = retriever(new_query_text, top_k)
        
        if not new_results or len(new_results[0]) == 0:
            # Fallback to initial results if re-retrieval fails
            new_doc_ids = initial_doc_ids
            new_doc_scores = initial_scores
        else:
            new_doc_ids = new_results[0]
            new_doc_scores = new_results[1]
        
        elapsed_time = time.time() - start_time
        
        return {
            'new_query_text': new_query_text,
            'new_doc_ids': new_doc_ids,
            'new_doc_scores': new_doc_scores.astype(np.float32) if isinstance(new_doc_scores, np.ndarray) else np.array(new_doc_scores, dtype=np.float32),
            'elapsed_time': elapsed_time,
        }
    
    def _extract_expansion_terms(self, original_query: str, segments: List[str]) -> List[str]:
        """
        Extract discriminative terms from top-k documents using TF-IDF.
        
        Args:
            original_query: the original query string
            segments: list of document text segments
            
        Returns:
            List of expansion terms (NUM_EXPANSION_TERMS most discriminative terms)
        """
        if not segments or all(not s for s in segments):
            return []
        
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 1),       # unigrams only
            min_df=1,                  # term must appear in ≥1 doc
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(segments)
        except ValueError:
            return []
        
        # Mean TF-IDF score across documents
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()
        
        # Sort by score, filter out query terms
        query_tokens = set(original_query.lower().split())
        ranked = sorted(
            zip(terms, mean_scores), key=lambda x: x[1], reverse=True
        )
        expansion_terms = [
            t for t, _ in ranked if t not in query_tokens
        ][:self.num_expansion_terms]
        
        return expansion_terms
    