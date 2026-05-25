from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ..core.agents import AgentBase
import os
import numpy as np
import time
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# system message sent to the LLM when generating query intents
SYSTEM_PROMPT = (
    "You are an information retrieval expert. "
    "Given a user query that may be general or vague, generate up to 3 possible interpretations or intents of the query. "
    "Each interpretation should represent a different aspect, meaning, or context of the query. "
    "Return ONLY the 3 interpretations, one per line, without numbering or explanations."
)
API_KEY = os.getenv("LLMAPI_KEY")
BASE_URL = os.getenv("BASE_URL_HPC")  # or BASE_URL_UNI depending on the environment
MODEL_NAME = os.getenv("MODEL_NAME_HPC")  # or MODEL_NAME_UNI depending on the environment

class IntentAgent(AgentBase):
    def __init__(self, embed_model: SentenceTransformer):
        super().__init__(agent_id=3, embed_model=embed_model)
        load_dotenv()
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "MAESTRO-Query-Intent",
                "X-Title": "Query Intent Expansion",
            }
        )
    
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate multiple query intents and retrieve documents for each.
        
        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'retriever': callable - retrieval function
                - 'top_k': int - number of documents per intent to retrieve
        Returns:
            Dict with:
                - 'new_doc_ids': List[str] - merged deduplicated document IDs from all intents, sorted by score (highest first)
                - 'new_scores': List[float] - aggregated scores for merged doc_ids (max score across intents, boosting docs found in multiple intents)
                - 'new_corpus': Dict - corpus for merged doc_ids
                - 'elapsed_time': float - time taken for processing
                - 'intents': List[str] - the generated query intents
        """
        original_query = query_features['query_text']
        retriever = query_features['retriever']
        top_k = query_features.get('top_k', 10)
        
        # Generate query intents from the original query
        start_time = time.time()
        intents = self._call_llm(original_query)
        
        # Retrieve documents for each intent and merge results
        all_doc_ids = []
        all_scores = []
        all_corpus = {}
        doc_scores_map = {}  # Track all scores per doc_id for aggregation
        doc_corpus_map = {}  # Track corpus for each doc_id
        
        for intent in intents:
            intent_doc_ids, intent_scores, intent_corpus = retriever(intent, top_k)
            
            # Collect scores for each doc_id (docs appearing in multiple intents get aggregated)
            for doc_id, score in zip(intent_doc_ids, intent_scores):
                if doc_id not in doc_scores_map:
                    doc_scores_map[doc_id] = []
                    if doc_id in intent_corpus:
                        doc_corpus_map[doc_id] = intent_corpus[doc_id]
                
                doc_scores_map[doc_id].append(score)
        
        # Aggregate scores: use max score (score boost for docs found in multiple intents)
        final_scores = {}
        for doc_id, scores in doc_scores_map.items():
            final_scores[doc_id] = max(scores)
        
        # Sort by final score (highest first)
        sorted_doc_ids = sorted(doc_scores_map.keys(), key=lambda d: final_scores[d], reverse=True)
        all_doc_ids = sorted_doc_ids
        all_scores = [final_scores[doc_id] for doc_id in sorted_doc_ids]
        all_corpus = doc_corpus_map
        
        elapsed_time = time.time() - start_time
        
        return {
            'new_doc_ids': all_doc_ids,
            'new_scores': all_scores,
            'new_corpus': all_corpus,
            'elapsed_time': elapsed_time,
            'intents': intents,
            'num_intents': len(intents),
            'num_unique_docs': len(all_doc_ids)
        }
    
    def _call_llm(self, query: str) -> List[str]:
        """
        Generate up to 3 possible interpretations/intents of the query.
        
        Args:
            query: Original query text
            
        Returns:
            List of query intents (up to 3)
        """
        user_message = (
            f"Query: {query}\n\n"
            f"Generate up to 3 possible interpretations or intents of this query."
        )
        
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.8,
            max_tokens=256,
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("No content returned from LLM")
        
        # Parse the response - each line is one intent
        intents = [line.strip() for line in content.strip().split('\n') if line.strip()]
        
        # Ensure we have at least one intent (fallback to original query)
        if not intents:
            intents = [query]
        
        return intents[:3]  # Limit to 3 intents
