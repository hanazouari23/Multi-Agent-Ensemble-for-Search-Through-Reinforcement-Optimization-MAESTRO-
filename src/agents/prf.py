
import time
from typing import Any, Dict

import numpy as np

from src.core.agents import AgentBase
from sklearn.feature_extraction.text import TfidfVectorizer


class PRFAgent(AgentBase):

    """Pseudo-Relevance Feedback (PRF) agent: Takes the top-k BM25 documents and extracts the most 
    discriminative terms from their content using TF-IDF weighting. These terms are concatenated with 
    the original query and BM25 is re-run, potentially surfacing documents the initial search missed. """

    def __init__(self, embed_model, num_expansion_terms=2):
        super().__init__(agent_id=2, embed_model=embed_model)
        self.num_expansion_terms = num_expansion_terms
    def compute_effects(self, query_features, raw_results) -> Dict[str, Any]:
        """
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
        retriever = query_features['retriever']
        original_query = query_features['query_text']

        start_time = time.time();
        # 1. Extract expansion terms from top-k documents using TF-IDF (placeholder logic)
        expansion_terms = self._extract_expansion_terms(original_query, raw_results)
        
        # 3. Reformulate query by appending expansion terms
        new_query_text = original_query + " " + " ".join(expansion_terms)
        
        # 4. Retrieve new documents with reformulated query
        new_raw_results = retriever(new_query_text)
        end_time = time.time()
        # 5. Extract doc_ids and scores from new results
        new_doc_ids =  new_raw_results[0] if new_raw_results else []
        new_doc_scores = new_raw_results[1] if new_raw_results else []
        new_segments = new_raw_results[2].values() if new_raw_results else []
        return {
            'new_query_text': new_query_text,
            'new_doc_ids': new_doc_ids,
            'new_doc_scores': new_doc_scores,
            'new_segments': new_segments,
            'elapsed_time': end_time - start_time, 
        }
    def _extract_expansion_terms(self, original_query, raw_results):
          
        # Placeholder: In a real implementation, compute TF-IDF scores across the top-k documents
        # and return the top N terms that are most discriminative
        segments = raw_results[2].values()  # Assuming raw_results[2] contains the corpus data

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 1),       # unigrams only — cleaner BM25 match query
            min_df=2,                  # term must appear in ≥2 docs to reduce noise
        )
        tfidf_matrix = vectorizer.fit_transform(segments)
    
        # Mean TF-IDF score across top-k docs
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()
    
        # Sort by score, filter query terms
        query_tokens = set(original_query.lower().split())
        ranked = sorted(
            zip(terms, mean_scores), key=lambda x: x[1], reverse=True
        )
        expansion_terms = [
            t for t, _ in ranked if t not in query_tokens
        ][:self.num_expansion_terms]
    
        return expansion_terms
    