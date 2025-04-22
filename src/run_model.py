from api.rag_system import RAGSystem
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run and save the RAG model for Kenyan Constitution')
    parser.add_argument('--load-saved', action='store_true', help='Load saved model instead of creating new one')
    args = parser.parse_args()
    
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag_system = RAGSystem(load_saved=args.load_saved)
    
    # Test the system with a sample query
    test_question = "What are the fundamental rights in the Kenyan Constitution?"
    print("\nTesting the system with a sample query:")
    print(f"Question: {test_question}")
    
    results = rag_system.query(test_question)
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   Distance: {result['distance']:.4f}")

if __name__ == "__main__":
    main() 