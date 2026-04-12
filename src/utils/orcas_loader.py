"""
ORCAS dataset loader utilities.

Converts ORCAS TSV format to query→clicked_docs index.
"""

from typing import Dict, List
from pathlib import Path


def load_orcas_tsv(filepath: str) -> Dict[str, List[str]]:
    """
    Load ORCAS TSV file into dict format expected by ClickPriorAgent.
    
    TSV format (tab-separated):
        query_id, query_text, doc_id, url
    
    Note: Any doc appearing in the file is considered clicked (binary signal).
    
    Returns:
        Dict[query_text] → List[doc_ids_with_clicks]
    
    Example:
        >>> orcas = load_orcas_tsv("data/orcas.tsv")
        >>> orcas["restaurants in passau"]
        ['D1265400', 'D3438005', 'D889000']
    """
    orcas_index: Dict[str, List[str]] = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            # Format: query_id, query_text, doc_id, url (url is optional)
            query_text = parts[1].lower().strip()
            doc_id = parts[2].strip()
            
            # Skip empty queries or docs
            if not query_text or not doc_id:
                continue
            
            # Add to index (presence in file means clicked)
            if query_text not in orcas_index:
                orcas_index[query_text] = []
            if doc_id not in orcas_index[query_text]:
                orcas_index[query_text].append(doc_id)
    
    return orcas_index


def load_orcas_tsv_sample(filepath: str, max_queries: int = 1000) -> Dict[str, List[str]]:
    """
    Load sample of ORCAS TSV for testing (to avoid loading entire 50MB file).
    
    Parameters
    ----------
    filepath : str
        Path to ORCAS TSV file
    max_queries : int
        Maximum unique queries to load
    
    Returns
    -------
    Dict[str, List[str]]
        Sampled ORCAS index: query_text → [doc_ids]
    """
    orcas_index: Dict[str, List[str]] = {}
    queries_seen = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(queries_seen) >= max_queries:
                break
            
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            # Format: query_id, query_text, doc_id, url
            query_text = parts[1].lower().strip()
            doc_id = parts[2].strip()
            
            # Skip empty queries or docs
            if not query_text or not doc_id:
                continue
            
            # Add to index
            queries_seen.add(query_text)
            if query_text not in orcas_index:
                orcas_index[query_text] = []
            if doc_id not in orcas_index[query_text]:
                orcas_index[query_text].append(doc_id)
    
    return orcas_index
