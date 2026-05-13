"""
Utility to collect MS MARCO queries with more than 5 qrels and generate CSV.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import ir_datasets

logger = logging.getLogger(__name__)


class QrelsCollector:
    """
    Collects MS MARCO queries with more than 5 qrels and exports to CSV.
    """
    
    def __init__(self, dataset_name: str = "msmarco-passage/train/judged"):
        """
        Initialize the qrels collector.
        
        Args:
            dataset_name: ir_datasets dataset name to load (default: msmarco-passage/train/judged)
        """
        self.dataset_name = dataset_name
        self.dataset = ir_datasets.load(dataset_name)
        self.qrels_dict: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
        self.query_text_dict: Dict[str, str] = {}
    
    def load_qrels(self) -> None:
        """
        Load all qrels from the dataset and organize by query_id.
        """
        logger.info(f"Loading qrels from {self.dataset_name}...")
        
        qrel_count = 0
        for qrel in self.dataset.qrels_iter():
            # qrel is a namedtuple with: query_id, doc_id, relevance, iteration
            query_id = str(qrel.query_id)
            doc_id = str(qrel.doc_id)
            relevance = int(qrel.relevance)
            iteration = str(qrel.iteration) if hasattr(qrel, 'iteration') else ""
            
            self.qrels_dict[query_id].append((doc_id, relevance, iteration))
            qrel_count += 1
        
        logger.info(f"Loaded {qrel_count} total qrels for {len(self.qrels_dict)} queries")
    
    def load_queries(self) -> None:
        """
        Load query texts from the dataset.
        """
        logger.info("Loading query texts...")
        
        query_count = 0
        for query in self.dataset.queries_iter():
            # query is a namedtuple with: query_id, text
            self.query_text_dict[str(query.query_id)] = query.text
            query_count += 1
        
        logger.info(f"Loaded {query_count} queries")
    
    def filter_queries(self, min_qrels: int = 5) -> Dict[str, List[Tuple[str, int, str]]]:
        """
        Filter queries that have at least min_qrels qrels.
        
        Args:
            min_qrels: Minimum number of qrels required for a query (default: 5)
        
        Returns:
            Dictionary of {query_id: [(doc_id, relevance, iteration), ...]}
        """
        filtered_qrels = {
            query_id: qrels
            for query_id, qrels in self.qrels_dict.items()
            if len(qrels) > min_qrels
        }
        
        logger.info(f"Filtered to {len(filtered_qrels)} queries with > {min_qrels} qrels")
        return filtered_qrels
    
    def export_to_csv(
        self,
        output_path: str,
        min_qrels: int = 5,
    ) -> None:
        """
        Export filtered qrels to CSV.
        
        Args:
            output_path: Path to output CSV file
            min_qrels: Minimum number of qrels required for a query (default: 5)
        """
        filtered_qrels = self.filter_queries(min_qrels)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(filtered_qrels)} queries to {output_path}")
        
        csv_rows = []
        for query_id, qrels_list in sorted(filtered_qrels.items()):
            for doc_id, relevance, iteration in qrels_list:
                csv_rows.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'relevance': relevance,
                    'iteration': iteration,
                })
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['query_id', 'doc_id', 'relevance', 'iteration']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        logger.info(f"Exported {len(csv_rows)} qrel entries to {output_path}")
    
    def get_stats(self, min_qrels: int = 5) -> Dict:
        """
        Get statistics about filtered qrels.
        
        Args:
            min_qrels: Minimum number of qrels required for a query (default: 5)
        
        Returns:
            Dictionary with statistics
        """
        filtered_qrels = self.filter_queries(min_qrels)
        
        total_qrel_entries = sum(len(qrels) for qrels in filtered_qrels.values())
        qrels_per_query = [len(qrels) for qrels in filtered_qrels.values()]
        
        stats = {
            'num_queries': len(filtered_qrels),
            'num_qrel_entries': total_qrel_entries,
            'min_qrels_per_query': min(qrels_per_query) if qrels_per_query else 0,
            'max_qrels_per_query': max(qrels_per_query) if qrels_per_query else 0,
            'avg_qrels_per_query': total_qrel_entries / len(filtered_qrels) if filtered_qrels else 0,
        }
        
        return stats


def collect_and_export_msmarco_qrels(
    output_csv: str = "data/msmarco_queries_5plus_qrels.csv",
    dataset_name: str = "msmarco-passage/train/judged",
    min_qrels: int = 5,
) -> None:
    """
    Main entry point: collect MS MARCO queries with > 5 qrels and export to CSV.
    
    Args:
        output_csv: Path to output CSV file
        dataset_name: ir_datasets dataset name to use
        min_qrels: Minimum number of qrels required for a query
    """
    collector = QrelsCollector(dataset_name=dataset_name)
    collector.load_qrels()
    collector.load_queries()
    collector.export_to_csv(output_csv, min_qrels=min_qrels)
    
    # Print statistics
    stats = collector.get_stats(min_qrels=min_qrels)
    print("\n" + "="*60)
    print("MS MARCO Qrels Collection Complete")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Output CSV: {output_csv}")
    print(f"Minimum qrels per query: {min_qrels}")
    print("-"*60)
    print(f"Number of queries: {stats['num_queries']}")
    print(f"Total qrel entries: {stats['num_qrel_entries']}")
    print(f"Qrels per query (min): {stats['min_qrels_per_query']}")
    print(f"Qrels per query (max): {stats['max_qrels_per_query']}")
    print(f"Qrels per query (avg): {stats['avg_qrels_per_query']:.2f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the collection
    collect_and_export_msmarco_qrels()
