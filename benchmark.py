import time
import statistics
from main import OptimizedRAGSystem

def benchmark_rag_system(test_queries, system_name="Optimized RAG"):
    """Benchmark the RAG system with a set of test queries."""
    print(f"\n🚀 Benchmarking {system_name}")
    print("=" * 50)
    
    # Initialize system
    start_time = time.time()
    rag_system = OptimizedRAGSystem()
    init_time = time.time() - start_time
    print(f"⏱️  System initialization: {init_time:.2f}s")
    
    # Test queries
    response_times = []
    context_counts = []
    similarities = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query[:50]}...")
        
        start_time = time.time()
        result = rag_system.process_query(query)
        query_time = time.time() - start_time
        
        response_times.append(query_time)
        context_counts.append(result['context_count'])
        
        if result['context_results']:
            similarities.append(result['context_results'][0]['similarity'])
        
        print(f"   ⏱️  Response time: {query_time:.2f}s")
        print(f"   📄 Context chunks: {result['context_count']}")
        if result['context_results']:
            print(f"   🎯 Best similarity: {result['context_results'][0]['similarity']:.3f}")
    
    # Calculate statistics
    avg_response_time = statistics.mean(response_times)
    median_response_time = statistics.median(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
    
    avg_context_count = statistics.mean(context_counts) if context_counts else 0
    avg_similarity = statistics.mean(similarities) if similarities else 0
    
    # Print summary
    print(f"\n📊 {system_name} Performance Summary")
    print("=" * 50)
    print(f"🔧 System initialization: {init_time:.2f}s")
    print(f"⏱️  Average response time: {avg_response_time:.2f}s")
    print(f"📊 Median response time: {median_response_time:.2f}s")
    print(f"⚡ Fastest response: {min_response_time:.2f}s")
    print(f"🐌 Slowest response: {max_response_time:.2f}s")
    print(f"📄 Average context chunks: {avg_context_count:.1f}")
    print(f"🎯 Average similarity score: {avg_similarity:.3f}")
    print(f"📈 Total queries processed: {len(test_queries)}")
    
    return {
        'init_time': init_time,
        'avg_response_time': avg_response_time,
        'median_response_time': median_response_time,
        'min_response_time': min_response_time,
        'max_response_time': max_response_time,
        'avg_context_count': avg_context_count,
        'avg_similarity': avg_similarity,
        'total_queries': len(test_queries)
    }

def main():
    """Run performance benchmarks."""
    # Test queries covering different aspects of the knowledge base
    test_queries = [
        "What is the mission of /dev/color?",
        "What is the A* Program?",
        "How does /dev/color collaborate with other organizations?",
        "What are the main sources of funding for /dev/color?",
        "What key achievements did /dev/color members accomplish in 2023?",
        "What types of events does /dev/color provide?",
        "How can individuals contribute to /dev/color?",
        "Which corporate partners supported /dev/color in 2023?",
        "What impact has the Executive Accelerator Program had?",
        "What are the key statistics on member engagement?"
    ]
    
    print("🎯 RAG System Performance Benchmark")
    print("=" * 60)
    
    # Run benchmark
    results = benchmark_rag_system(test_queries)
    
    # Performance recommendations
    print(f"\n💡 Performance Recommendations:")
    print("=" * 40)
    
    if results['avg_response_time'] > 2.0:
        print("⚠️  Response times are slow. Consider:")
        print("   - Using a faster embedding model")
        print("   - Implementing more aggressive caching")
        print("   - Reducing chunk size")
    
    if results['avg_similarity'] < 0.7:
        print("⚠️  Low similarity scores. Consider:")
        print("   - Adjusting similarity threshold")
        print("   - Improving text chunking strategy")
        print("   - Using a better embedding model")
    
    if results['avg_context_count'] < 1:
        print("⚠️  Few context chunks retrieved. Consider:")
        print("   - Lowering similarity threshold")
        print("   - Increasing TOP_K_RESULTS")
        print("   - Improving chunk quality")
    
    print(f"\n✅ Benchmark completed successfully!")
    print(f"📈 Overall performance score: {calculate_performance_score(results):.1f}/10")

def calculate_performance_score(results):
    """Calculate a performance score from 1-10."""
    score = 10
    
    # Penalize slow response times
    if results['avg_response_time'] > 3.0:
        score -= 3
    elif results['avg_response_time'] > 2.0:
        score -= 2
    elif results['avg_response_time'] > 1.0:
        score -= 1
    
    # Penalize low similarity scores
    if results['avg_similarity'] < 0.6:
        score -= 2
    elif results['avg_similarity'] < 0.7:
        score -= 1
    
    # Penalize too few context chunks
    if results['avg_context_count'] < 1:
        score -= 2
    
    return max(1, score)

if __name__ == "__main__":
    main() 