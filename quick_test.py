#!/usr/bin/env python3
"""
Quick test to verify current retrieval system works.
"""

import sys
import json
sys.path.append('src')

from embed_index import EmbeddingIndexer
from ingest import DataIngester

def test_current_system():
    print("🧪 TESTING CURRENT SANSKRIT TUTOR SYSTEM")
    print("=" * 50)
    
    # 1. Test data loading
    print("\n1️⃣ Testing data loading...")
    try:
        ingester = DataIngester(".")
        passages = ingester.load_passages("user_assets/passages.jsonl")
        qa_pairs = ingester.load_qa_pairs("user_assets/qa_pairs.jsonl", {p.id for p in passages})
        
        print(f"✅ Loaded {len(passages)} passages")
        print(f"✅ Loaded {len(qa_pairs)} QA pairs")
        
        # Show sample data
        sample_passage = passages[0]
        print(f"\n📖 Sample passage:")
        print(f"   ID: {sample_passage.id}")
        print(f"   Work: {sample_passage.work}")
        print(f"   Devanagari: {sample_passage.text_devanagari[:50]}...")
        print(f"   IAST: {sample_passage.text_iast[:50]}...")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # 2. Test indexer
    print("\n2️⃣ Testing indexer...")
    try:
        # Load config to get embedding model name
        import yaml
        with open("user_assets/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        embeddings_model = config.get('embeddings_model', 'sentence-transformers/all-mpnet-base-v2')
        indexer = EmbeddingIndexer(embeddings_model, ".")
        
        # Load embedding model
        if not indexer.load_embedding_model():
            print("❌ Failed to load embedding model")
            return False
        
        print(f"✅ Loaded embedding model: {indexer.embeddings_model_name}")
        print(f"   Dimension: {indexer.dimension}")
        
        # Load existing index
        index, metadata = indexer.load_faiss_index("data/faiss.index")
        print(f"✅ Loaded FAISS index: {index.ntotal} vectors")
        print(f"   Model used: {metadata.get('embeddings_model', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Indexer test failed: {e}")
        return False
    
    # 3. Test search functionality
    print("\n3️⃣ Testing search functionality...")
    try:
        test_queries = [
            "योगः कर्मसु कौशलम्",  # Devanagari
            "yoga action skill",     # English
            "dharma duty",           # English concept
            "कृष्ण अर्जुन",            # Devanagari names
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = indexer.search(index, metadata, query, k=3)
            
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                passage = result['passage']
                score = result['score']
                print(f"   {i}. [{passage['id']}] Score: {score:.3f}")
                print(f"      {passage['text_devanagari'][:60]}...")
                print(f"      {passage['work']} {passage['chapter']}.{passage['verse']}")
        
        print(f"\n✅ Search functionality working!")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False
    
    # 4. Test a few QA pairs for retrieval quality
    print("\n4️⃣ Testing retrieval quality...")
    try:
        correct_retrievals = 0
        total_tests = 0
        
        # Test first 10 QA pairs
        for qa in qa_pairs[:10]:
            total_tests += 1
            results = indexer.search(index, metadata, qa.question, k=5)
            retrieved_ids = [r['passage_id'] for r in results]
            
            # Check if any gold passage is in top-5
            if any(gold_id in retrieved_ids for gold_id in qa.related_passage_ids):
                correct_retrievals += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"   {status} Q: {qa.question[:50]}...")
            print(f"      Gold: {qa.related_passage_ids}")
            print(f"      Retrieved: {retrieved_ids[:3]}")
        
        accuracy = correct_retrievals / total_tests
        print(f"\n📊 Retrieval Accuracy: {accuracy:.2f} ({correct_retrievals}/{total_tests})")
        
    except Exception as e:
        print(f"❌ Retrieval quality test failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("\nYour Sanskrit Tutor system is working correctly!")
    print("\nNext steps:")
    print("1. Run: python test_embedding_model.py")
    print("2. Launch UI: python src/ui_gradio.py --config user_assets/config.yaml")
    print("3. Try Sanskrit chat: python src/sanskrit_chat_ui.py --config user_assets/config.yaml")
    
    return True

if __name__ == "__main__":
    test_current_system()
