#!/usr/bin/env python3
"""
Script to collect MS MARCO queries with more than 5 qrels and generate CSV.

Usage:
    python src/collect_qrels.py [--output OUTPUT_CSV] [--min-qrels MIN_QRELS]

Example:
    python src/collect_qrels.py --output data/msmarco_qrels.csv --min-qrels 5
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.qrels_collector import QrelsCollector


def main():
    parser = argparse.ArgumentParser(
        description="Collect MS MARCO queries with more than N qrels and export to CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/msmarco_queries_5plus_qrels.csv",
        help="Output CSV file path (default: data/msmarco_queries_5plus_qrels.csv)"
    )
    parser.add_argument(
        "--min-qrels",
        type=int,
        default=5,
        help="Minimum number of qrels per query (default: 5)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco-passage/train/judged",
        help="ir_datasets dataset name (default: msmarco-passage/train/judged)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting MS MARCO qrels collection...")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Minimum qrels: {args.min_qrels}")
    logger.info(f"Dataset: {args.dataset}")
    
    try:
        # Create collector
        collector = QrelsCollector(dataset_name=args.dataset)
        
        # Load qrels and queries
        collector.load_qrels()
        collector.load_queries()
        
        # Export to CSV
        collector.export_to_csv(args.output, min_qrels=args.min_qrels)
        
        # Print statistics
        stats = collector.get_stats(min_qrels=args.min_qrels)
        
        print("\n" + "="*70)
        print("MS MARCO Qrels Collection Complete")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Output CSV: {args.output}")
        print(f"Minimum qrels per query: {args.min_qrels}")
        print("-"*70)
        print(f"Number of queries with > {args.min_qrels} qrels: {stats['num_queries']}")
        print(f"Total qrel entries exported: {stats['num_qrel_entries']}")
        print(f"Qrels per query (min): {stats['min_qrels_per_query']}")
        print(f"Qrels per query (max): {stats['max_qrels_per_query']}")
        print(f"Qrels per query (avg): {stats['avg_qrels_per_query']:.2f}")
        print("="*70 + "\n")
        
        logger.info("Collection completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during collection: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
