"""
Test Script for RAG System (Book-Based Learning)
"""
import os
from pathlib import Path
from src.tools.vector_store import VectorStoreManager
from src.agents.rag_agent import RAGAgent
from config.settings import settings
from src.utils.logger import app_logger


def create_sample_documents():
    """Create sample math documents for testing"""
    
    sample_docs = {
        "algebra_basics.txt": """
# Algebra Basics

## Linear Equations
A linear equation is an equation of the form: $ax + b = c$

To solve for $x$:
1. Subtract $b$ from both sides: $ax = c - b$
2. Divide both sides by $a$: $x = \\frac{c - b}{a}$

Example: Solve $2x + 5 = 15$
- Subtract 5: $2x = 10$
- Divide by 2: $x = 5$

## Quadratic Equations
A quadratic equation has the form: $ax^2 + bx + c = 0$

The quadratic formula is:
$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

Example: Solve $x^2 - 5x + 6 = 0$
Using the formula with $a=1, b=-5, c=6$:
$$x = \\frac{5 \\pm \\sqrt{25 - 24}}{2} = \\frac{5 \\pm 1}{2}$$

Solutions: $x = 3$ or $x = 2$
""",
        
        "calculus_intro.txt": """
# Introduction to Calculus

## Derivatives
The derivative measures the rate of change of a function.

### Basic Rules
1. Power Rule: $\\frac{d}{dx}(x^n) = nx^{n-1}$
2. Constant Rule: $\\frac{d}{dx}(c) = 0$
3. Sum Rule: $\\frac{d}{dx}(f + g) = f' + g'$
4. Product Rule: $\\frac{d}{dx}(fg) = f'g + fg'$

### Examples
1. $\\frac{d}{dx}(x^3) = 3x^2$
2. $\\frac{d}{dx}(5x^2 + 3x - 7) = 10x + 3$
3. $\\frac{d}{dx}(x^2 \\cdot x^3) = 2x \\cdot x^3 + x^2 \\cdot 3x^2 = 5x^4$

## Integrals
Integration is the reverse of differentiation.

$$\\int x^n dx = \\frac{x^{n+1}}{n+1} + C$$

Example: $\\int (3x^2 + 2x) dx = x^3 + x^2 + C$
""",
        
        "geometry_formulas.txt": """
# Geometry Formulas

## Circle
- Circumference: $C = 2\\pi r$
- Area: $A = \\pi r^2$

Example: A circle with radius $r = 5$
- Circumference: $C = 2\\pi(5) = 10\\pi \\approx 31.42$
- Area: $A = \\pi(5)^2 = 25\\pi \\approx 78.54$

## Triangle
- Area: $A = \\frac{1}{2}bh$
- Perimeter: $P = a + b + c$

## Rectangle
- Area: $A = l \\times w$
- Perimeter: $P = 2(l + w)$

## Pythagorean Theorem
For a right triangle:
$$a^2 + b^2 = c^2$$

where $c$ is the hypotenuse.

Example: If $a = 3$ and $b = 4$
$$c = \\sqrt{3^2 + 4^2} = \\sqrt{9 + 16} = \\sqrt{25} = 5$$
""",
        
        "statistics_basics.txt": """
# Statistics Basics

## Measures of Central Tendency

### Mean (Average)
$$\\mu = \\frac{1}{n}\\sum_{i=1}^{n} x_i$$

Example: For data [5, 8, 12, 15, 20]
$$\\mu = \\frac{5 + 8 + 12 + 15 + 20}{5} = \\frac{60}{5} = 12$$

### Median
The middle value when data is ordered.

### Mode
The most frequently occurring value.

## Measures of Spread

### Standard Deviation
$$\\sigma = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(x_i - \\mu)^2}$$

### Variance
$$\\sigma^2 = \\frac{1}{n}\\sum_{i=1}^{n}(x_i - \\mu)^2$$

## Probability
$$P(A) = \\frac{\\text{Number of favorable outcomes}}{\\text{Total number of outcomes}}$$

Example: Probability of rolling a 6 on a die = $\\frac{1}{6}$
"""
    }
    
    # Create sample documents
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    created_files = []
    for filename, content in sample_docs.items():
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(filepath)
        print(f"✅ Created: {filename}")
    
    return created_files


def test_vector_store():
    """Test vector store functionality"""
    print("\n" + "="*60)
    print("TESTING VECTOR STORE")
    print("="*60 + "\n")
    
    # Initialize vector store
    vector_store = VectorStoreManager()
    
    # Create sample documents
    print("📝 Creating sample documents...")
    sample_files = create_sample_documents()
    
    # Add documents to vector store
    print("\n📤 Adding documents to vector store...")
    for filepath in sample_files:
        filename = os.path.basename(filepath)
        category = filename.split('_')[0]  # Get category from filename
        
        result = vector_store.add_documents(filepath, category)
        
        if result["success"]:
            print(f"✅ {filename}: {result['chunks_added']} chunks added")
        else:
            print(f"❌ {filename}: {result['message']}")
    
    # Get statistics
    print("\n📊 Vector Store Statistics:")
    stats = vector_store.get_collection_stats()
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Unique Documents: {stats['unique_documents']}")
    print(f"Categories: {', '.join(stats['categories'])}")
    print(f"Documents: {', '.join(stats['document_names'])}")
    
    # Test search
    print("\n🔍 Testing search functionality...")
    queries = [
        "What is the quadratic formula?",
        "How do you find the derivative of x^3?",
        "What is the area of a circle?",
        "How do you calculate mean?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_store.search(query, n_results=2)
        
        if results:
            print(f"Found {len(results)} results:")
            for idx, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'Unknown')
                print(f"  {idx}. Source: {source}")
                print(f"     Text: {result['text'][:100]}...")
        else:
            print("No results found")
    
    return vector_store


def test_rag_agent(vector_store):
    """Test RAG agent functionality"""
    print("\n" + "="*60)
    print("TESTING RAG AGENT")
    print("="*60 + "\n")
    
    # Initialize RAG agent
    rag_agent = RAGAgent()
    
    # Test questions
    questions = [
        {
            "question": "Explain the quadratic formula and give an example",
            "category": "algebra"
        },
        {
            "question": "What is the power rule in calculus?",
            "category": "calculus"
        },
        {
            "question": "How do you calculate the area of a circle with radius 5?",
            "category": "geometry"
        },
        {
            "question": "Explain how to calculate the mean of a dataset",
            "category": "statistics"
        },
        {
            "question": "Solve a linear equation step by step",
            "category": None  # Test without category filter
        }
    ]
    
    for idx, test in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {idx}: {test['question']}")
        if test['category']:
            print(f"Category: {test['category']}")
        print('='*60)
        
        result = rag_agent.answer_question(
            question=test['question'],
            category=test['category'],
            n_context=3
        )
        
        if result["success"]:
            print(f"\n✅ Answer:")
            print(result['answer'])
            print(f"\n📚 Sources: {', '.join(result['sources'])}")
            print(f"🔢 Context Chunks Used: {result['context_used']}")
            print(f"🤖 Model: {result['model_used']}")
        else:
            print(f"\n❌ Error: {result['answer']}")
    
    # Test additional features
    print("\n" + "="*60)
    print("TESTING ADDITIONAL FEATURES")
    print("="*60 + "\n")
    
    # Test explain concept
    print("📖 Explaining concept: derivatives")
    result = rag_agent.explain_concept("derivatives", category="calculus")
    if result["success"]:
        print(f"Answer: {result['answer'][:200]}...")
    
    # Test find examples
    print("\n💡 Finding examples for: quadratic equations")
    result = rag_agent.find_examples("quadratic equations", category="algebra")
    if result["success"]:
        print(f"Answer: {result['answer'][:200]}...")
    
    # Test summarize topic
    print("\n📝 Summarizing topic: geometry formulas")
    result = rag_agent.summarize_topic("geometry formulas", category="geometry")
    if result["success"]:
        print(f"Answer: {result['answer'][:200]}...")


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("RAG SYSTEM TEST SUITE")
    print("Book-Based Learning Implementation")
    print("="*60)
    
    try:
        # Test vector store
        vector_store = test_vector_store()
        
        # Test RAG agent
        test_rag_agent(vector_store)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        print("📚 To use the Book-Based Learning system:")
        print("   Run: streamlit run book_learning.py")
        print("\n🧮 To use the Math Agent:")
        print("   Run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
