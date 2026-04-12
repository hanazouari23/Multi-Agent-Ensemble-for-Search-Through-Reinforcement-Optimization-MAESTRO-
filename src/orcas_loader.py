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
        query_id, query_text, doc_id, click_signal, ...
    
    Returns:
        Dict[query_text] → List[doc_ids_with_clicks]
    
    Example:
        >>> orcas = load_orcas_tsv("data/orcas.tsv")
        >>> orcas["restaurants in passau"]
        ['doc1', 'doc3', 'doc5']
    """
    orcas_index: Dict[str, List[str]] = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                # Skip header if present
                if line.startswith("query") or line.startswith("#"):
                    continue
            
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            # Assuming format: query_id, query_text, doc_id, click_signal, ...
            query_text = parts[1].lower().strip()
            doc_id = parts[2].strip()
            
            # Only include if there's a click signal (non-zero)
            if len(parts) > 3 and parts[3].strip():
                try:
                    click_signal = float(parts[3])
                    if click_signal > 0:
                        if query_text not in orcas_index:
                            orcas_index[query_text] = []
                        if doc_id not in orcas_index[query_text]:
                            orcas_index[query_text].append(doc_id)
                except (ValueError, IndexError):
                    pass
    
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
        Sampled ORCAS index
    """
    orcas_index: Dict[str, List[str]] = {}
    queries_seen = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(queries_seen) >= max_queries:
                break
            
            if i == 0 and (line.startswith("query") or line.startswith("#")):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            query_text = parts[1].lower().strip()
            doc_id = parts[2].strip()
            
            if len(parts) > 3 and parts[3].strip():
                try:
                    click_signal = float(parts[3])
                    if click_signal > 0:
                        queries_seen.add(query_text)
                        if query_text not in orcas_index:
                            orcas_index[query_text] = []
                        if doc_id not in orcas_index[query_text]:
                            orcas_index[query_text].append(doc_id)
                except (ValueError, IndexError):
                    pass
    
    return orcas_index
