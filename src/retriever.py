import requests
from typing import List, Dict, Any
import os

class Retriever:
    """
    Retriever that accesses OpenSearch endpoint to retrieve top 50 documents
    from the msmarco-v2.1-segmented index using BM25 matching.
    """
    
    def __init__(self, endpoint: str = "https://opensearch.pads.fim.uni-passau.de/msmarco-v2.1-segmented/_search",
                 username: str = "hana",
                 password: str = "4DBpreroRMc!fPPczxaJ",
                 index_field: str = "segment",
                 top_k: int = 50):
        """
        Initialize the retriever with OpenSearch credentials and parameters.
        
        Args:
            endpoint: OpenSearch endpoint URL
            username: Username for authentication
            password: Password for authentication
            index_field: Field to search in (default: "segment")
            top_k: Number of top documents to retrieve (default: 50)
        """
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.index_field = index_field
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top documents for a given query using BM25 matching.
        
        Args:
            query: The search query string
            top_k: Number of top documents to retrieve (overrides instance default if provided)
        
        Returns:
            List of documents with id, text (segment), and score
        """
        if top_k is None:
            top_k = self.top_k
        
        # Construct OpenSearch query
        search_body = {
            "query": {
                "match": {
                    self.index_field: query
                }
            },
            "size": top_k
        }
        
        try:
            # Make authenticated request to OpenSearch
            response = requests.get(
                self.endpoint,
                json=search_body,
                auth=(self.username, self.password),
                headers={"Content-Type": "application/json"},
                verify=False  # Note: Set to True in production with proper certificates
            )
            response.raise_for_status()
            
            # Parse results
            results = response.json()
            documents = []
            
            if "hits" in results and "hits" in results["hits"]:
                for hit in results["hits"]["hits"]:
                    doc = {
                        "id": hit.get("_id"),
                        "text": hit.get("_source", {}).get(self.index_field, ""),
                        "score": hit.get("_score", 0.0),
                        "source": hit.get("_source", {})  # Full document source
                    }
                    documents.append(doc)
            
            return documents
        
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving documents: {e}")
            return []


# Example usage
if __name__ == "__main__":
    retriever = Retriever()
    
    # Test query
    query = "Restaurants in Passau"
    results = retriever.retrieve(query)
    
    print(f"Retrieved {len(results)} documents for query: '{query}'")
    for i, doc in enumerate(results[:5], 1):  # Print top 5
        print(f"\n{i}. Score: {doc['score']:.4f}")
        print(f"   ID: {doc['id']}")
        print(f"   Text: {doc['text'][:100]}...")
