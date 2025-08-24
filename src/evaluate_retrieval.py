#!/usr/bin/env python3
"""
Retrieval Quality Evaluation for Sanskrit Tutor.
Measures Recall@k, MRR, and citation accuracy to compare embedding models.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import argparse
import random

try:
    from .embed_index import EmbeddingIndexer
    from .ingest import DataIngester
except ImportError:
    from embed_index import EmbeddingIndexer
    from ingest import DataIngester


@dataclass
class EvaluationQuery:
    """A query with ground truth for evaluation."""
    query: str
    script: str  # 'devanagari', 'iast', 'english'
    gold_passage_ids: List[str]
    difficulty: str
    query_type: str  # 'factual', 'conceptual', 'translation'


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.ingester = DataIngester(".")
        self.indexer = EmbeddingIndexer(".")
        
        # Load data
        self.passages = {}
        self.qa_pairs = {}
        
        # Index and metadata (loaded when needed)
        self.index = None
        self.metadata = None
        
    def load_data(self) -> bool:
        """Load passages and QA pairs."""
        try:
            print("Loading evaluation data...")
            
            # Load config to get file paths
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            passages_file = config.get('passages_file', 'user_assets/passages.jsonl')
            qa_file = config.get('qa_file', 'user_assets/qa_pairs.jsonl')
            
            # Load passages
            passages_list = self.ingester.load_passages(passages_file)
            self.passages = {p.id: p for p in passages_list}
            print(f"✓ Loaded {len(self.passages)} passages")
            
            # Create valid passage IDs set
            valid_passage_ids = {p.id for p in passages_list}
            
            # Load QA pairs
            qa_list = self.ingester.load_qa_pairs(qa_file, valid_passage_ids)
            self.qa_pairs = {qa.id: qa for qa in qa_list}
            print(f"✓ Loaded {len(self.qa_pairs)} QA pairs")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            return False
    
    def create_evaluation_queries(self, num_samples: int = 200) -> List[EvaluationQuery]:
        """Create evaluation queries from QA pairs with ground truth."""
        queries = []
        
        # Sample QA pairs for different scripts and difficulties
        qa_list = list(self.qa_pairs.values())
        random.shuffle(qa_list)
        
        for qa in qa_list[:num_samples]:
            # Extract Sanskrit terms from question for Devanagari queries
            devanagari_query = self._extract_sanskrit_terms(qa.question)
            
            # Create multiple query variants
            query_variants = [
                EvaluationQuery(
                    query=qa.question,
                    script='english',
                    gold_passage_ids=qa.related_passage_ids,
                    difficulty=qa.difficulty,
                    query_type='factual'
                )
            ]
            
            # Add Sanskrit variant if we found terms
            if devanagari_query:
                query_variants.append(EvaluationQuery(
                    query=devanagari_query,
                    script='devanagari',
                    gold_passage_ids=qa.related_passage_ids,
                    difficulty=qa.difficulty,
                    query_type='translation'
                ))
            
            queries.extend(query_variants)
        
        return queries[:num_samples]
    
    def _extract_sanskrit_terms(self, question: str) -> str:
        """Extract Sanskrit terms from English questions for Devanagari queries."""
        # Simple heuristic: look for Sanskrit terms in the question
        sanskrit_patterns = [
            "dharma", "yoga", "karma", "moksha", "atman", "brahman",
            "Krishna", "Arjuna", "Bhagavad", "Gita"
        ]
        
        # Map to Devanagari equivalents
        sanskrit_map = {
            "dharma": "धर्म",
            "yoga": "योग", 
            "karma": "कर्म",
            "moksha": "मोक्ष",
            "atman": "आत्मा",
            "brahman": "ब्रह्म",
            "Krishna": "कृष्ण",
            "Arjuna": "अर्जुन"
        }
        
        for eng_term in sanskrit_patterns:
            if eng_term.lower() in question.lower():
                return sanskrit_map.get(eng_term, eng_term)
        
        return ""
    
    def evaluate_index(self, index_path: str, queries: List[EvaluationQuery], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate retrieval quality for a given index."""
        try:
            # Load index
            print(f"Loading index: {index_path}")
            if not self.indexer.load_index(index_path):
                raise RuntimeError(f"Failed to load index: {index_path}")
            
            results = {
                'total_queries': len(queries),
                'recall_at_k': {k: 0.0 for k in k_values},
                'mrr': 0.0,
                'by_script': {},
                'by_difficulty': {},
                'query_failures': []
            }
            
            reciprocal_ranks = []
            
            for query in queries:
                try:
                    # Perform search
                    search_results = self.indexer.search(query.query, k=max(k_values))
                    retrieved_ids = [r['passage_id'] for r in search_results]
                    
                    # Calculate recall@k for this query
                    gold_set = set(query.gold_passage_ids)
                    
                    for k in k_values:
                        retrieved_k = set(retrieved_ids[:k])
                        if gold_set.intersection(retrieved_k):
                            results['recall_at_k'][k] += 1
                    
                    # Calculate reciprocal rank
                    rr = 0.0
                    for rank, passage_id in enumerate(retrieved_ids, 1):
                        if passage_id in gold_set:
                            rr = 1.0 / rank
                            break
                    reciprocal_ranks.append(rr)
                    
                    # Track by script and difficulty
                    script = query.script
                    difficulty = query.difficulty
                    
                    if script not in results['by_script']:
                        results['by_script'][script] = {'total': 0, 'hits': 0}
                    if difficulty not in results['by_difficulty']:
                        results['by_difficulty'][difficulty] = {'total': 0, 'hits': 0}
                    
                    results['by_script'][script]['total'] += 1
                    results['by_difficulty'][difficulty]['total'] += 1
                    
                    if rr > 0:
                        results['by_script'][script]['hits'] += 1
                        results['by_difficulty'][difficulty]['hits'] += 1
                
                except Exception as e:
                    results['query_failures'].append({
                        'query': query.query,
                        'error': str(e)
                    })
            
            # Finalize metrics
            total_queries = len(queries)
            for k in k_values:
                results['recall_at_k'][k] = results['recall_at_k'][k] / total_queries
            
            results['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
            
            # Calculate success rates by category
            for category in ['by_script', 'by_difficulty']:
                for key, stats in results[category].items():
                    if stats['total'] > 0:
                        stats['success_rate'] = stats['hits'] / stats['total']
                    else:
                        stats['success_rate'] = 0.0
            
            return results
            
        except Exception as e:
            print(f"ERROR: Evaluation failed: {e}")
            return {}
    
    def compare_indices(self, index1_path: str, index2_path: str, 
                       queries: List[EvaluationQuery]) -> Dict[str, Any]:
        """Compare two indices side by side."""
        print("=" * 60)
        print("RETRIEVAL QUALITY COMPARISON")
        print("=" * 60)
        
        # Evaluate first index
        print(f"\n📊 Evaluating Index 1: {Path(index1_path).name}")
        results1 = self.evaluate_index(index1_path, queries)
        
        # Evaluate second index  
        print(f"\n📊 Evaluating Index 2: {Path(index2_path).name}")
        results2 = self.evaluate_index(index2_path, queries)
        
        # Compare results
        comparison = {
            'index1': {
                'path': index1_path,
                'results': results1
            },
            'index2': {
                'path': index2_path, 
                'results': results2
            },
            'improvements': {}
        }
        
        # Calculate improvements
        if results1 and results2:
            for k in [1, 3, 5, 10]:
                r1 = results1['recall_at_k'][k]
                r2 = results2['recall_at_k'][k]
                improvement = ((r2 - r1) / r1 * 100) if r1 > 0 else 0
                comparison['improvements'][f'recall_at_{k}'] = improvement
            
            mrr1 = results1['mrr']
            mrr2 = results2['mrr']
            mrr_improvement = ((mrr2 - mrr1) / mrr1 * 100) if mrr1 > 0 else 0
            comparison['improvements']['mrr'] = mrr_improvement
        
        return comparison
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print detailed evaluation report."""
        if not results:
            print("No results to display")
            return
        
        print(f"\n📈 RETRIEVAL QUALITY METRICS")
        print(f"Total queries: {results['total_queries']}")
        print(f"Query failures: {len(results['query_failures'])}")
        
        print(f"\n🎯 Recall@k:")
        for k, recall in results['recall_at_k'].items():
            print(f"  Recall@{k}: {recall:.3f}")
        
        print(f"\n🔄 Mean Reciprocal Rank: {results['mrr']:.3f}")
        
        print(f"\n📝 By Script:")
        for script, stats in results['by_script'].items():
            print(f"  {script}: {stats['success_rate']:.3f} ({stats['hits']}/{stats['total']})")
        
        print(f"\n📊 By Difficulty:")
        for difficulty, stats in results['by_difficulty'].items():
            print(f"  {difficulty}: {stats['success_rate']:.3f} ({stats['hits']}/{stats['total']})")
        
        if results['query_failures']:
            print(f"\n❌ Failed Queries ({len(results['query_failures'])}):")
            for failure in results['query_failures'][:5]:  # Show first 5
                print(f"  - {failure['query'][:50]}... → {failure['error']}")
    
    def print_comparison_report(self, comparison: Dict[str, Any]):
        """Print comparison report between two indices."""
        r1 = comparison['index1']['results']
        r2 = comparison['index2']['results']
        improvements = comparison['improvements']
        
        print(f"\n📊 COMPARISON RESULTS")
        print(f"Index 1: {Path(comparison['index1']['path']).name}")
        print(f"Index 2: {Path(comparison['index2']['path']).name}")
        
        print(f"\n🎯 Recall@k Comparison:")
        for k in [1, 3, 5, 10]:
            r1_val = r1['recall_at_k'][k]
            r2_val = r2['recall_at_k'][k]
            improvement = improvements[f'recall_at_{k}']
            
            indicator = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"  Recall@{k}: {r1_val:.3f} → {r2_val:.3f} {indicator} ({improvement:+.1f}%)")
        
        print(f"\n🔄 MRR Comparison:")
        mrr_improvement = improvements['mrr']
        mrr_indicator = "📈" if mrr_improvement > 0 else "📉" if mrr_improvement < 0 else "➡️"
        print(f"  MRR: {r1['mrr']:.3f} → {r2['mrr']:.3f} {mrr_indicator} ({mrr_improvement:+.1f}%)")
        
        # Summary recommendation
        if mrr_improvement > 5:
            print(f"\n✅ RECOMMENDATION: Index 2 shows significant improvement ({mrr_improvement:.1f}% MRR gain)")
        elif mrr_improvement > 0:
            print(f"\n📝 RECOMMENDATION: Index 2 shows modest improvement ({mrr_improvement:.1f}% MRR gain)")
        else:
            print(f"\n⚠️  RECOMMENDATION: Index 1 performs better or equivalent")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--index1", help="First index to evaluate/compare")
    parser.add_argument("--index2", help="Second index to compare (optional)")
    parser.add_argument("--queries", type=int, default=200, help="Number of evaluation queries")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(args.config)
    if not evaluator.load_data():
        print("Failed to load data")
        return
    
    # Create evaluation queries
    print(f"\nCreating {args.queries} evaluation queries...")
    eval_queries = evaluator.create_evaluation_queries(args.queries)
    print(f"✓ Created {len(eval_queries)} evaluation queries")
    
    # Categorize queries
    script_counts = {}
    difficulty_counts = {}
    for q in eval_queries:
        script_counts[q.script] = script_counts.get(q.script, 0) + 1
        difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
    
    print(f"\nQuery breakdown:")
    print(f"  By script: {script_counts}")
    print(f"  By difficulty: {difficulty_counts}")
    
    if args.index2:
        # Compare two indices
        print(f"\n🔍 COMPARING TWO INDICES")
        comparison = evaluator.compare_indices(args.index1, args.index2, eval_queries)
        evaluator.print_comparison_report(comparison)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")
    
    else:
        # Evaluate single index
        index_path = args.index1 or "data/faiss.index"
        print(f"\n🔍 EVALUATING SINGLE INDEX: {index_path}")
        
        results = evaluator.evaluate_index(index_path, eval_queries)
        evaluator.print_evaluation_report(results)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
