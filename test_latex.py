"""
Test script to verify LaTeX rendering in solutions
"""
from src.agents.orchestrator import MathAgentOrchestrator
from src.utils.logger import app_logger


def test_latex_rendering():
    """Test LaTeX rendering with sample problems"""
    
    print("=" * 70)
    print("Testing LaTeX Rendering in Math Solutions")
    print("=" * 70)
    
    orchestrator = MathAgentOrchestrator()
    
    # Test problems for different categories
    test_problems = [
        {
            "problem": "Solve for x: 2x + 5 = 15",
            "category": "Algebra"
        },
        {
            "problem": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
            "category": "Calculus"
        },
        {
            "problem": "Find the area of a circle with radius 7 cm",
            "category": "Geometry"
        }
    ]
    
    for idx, test in enumerate(test_problems, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {idx}: {test['category']}")
        print(f"{'=' * 70}")
        print(f"Problem: {test['problem']}")
        print(f"\nProcessing...")
        
        try:
            result = orchestrator.process_problem(test['problem'])
            
            print(f"\nCategory Detected: {result['routing']['category']}")
            print(f"\nSOLUTION:\n")
            print(result['solution']['solution'])
            print(f"\n{'=' * 70}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print("\nNote: LaTeX formulas are wrapped in $ (inline) or $$ (display).")
    print("They will render properly in Streamlit, Jupyter, or Markdown viewers.")


if __name__ == "__main__":
    test_latex_rendering()
