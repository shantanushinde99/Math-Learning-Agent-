"""
Main CLI interface for the Math Routing Agent
"""
from src.agents.orchestrator import MathAgentOrchestrator
from src.utils.logger import app_logger
from config.settings import settings


def main():
    """Main CLI function"""
    print("=" * 60)
    print("🧮 Math Routing Agent - CLI Interface")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = MathAgentOrchestrator()
    
    print(f"\nUsing Model: {settings.LLM_MODEL}")
    print(f"Provider: {settings.LLM_PROVIDER.upper()}")
    print("\nAvailable Models:")
    print("Groq:", ", ".join(settings.GROQ_MODELS))
    print("Gemini:", ", ".join(settings.GEMINI_MODELS))
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Solve a math problem")
        print("2. View feedback stats")
        print("3. View learning insights")
        print("4. Change model")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            problem = input("\nEnter your math problem: ").strip()
            if problem:
                print("\n" + "=" * 60)
                result = orchestrator.process_problem(problem)
                
                print(f"\n🧭 Category: {result['routing']['category'].upper()}")
                print(f"📝 Reasoning: {result['routing']['reasoning']}")
                print("\n✅ SOLUTION:")
                print("-" * 60)
                print(result['solution']['solution'])
                print("=" * 60)
                
                # Collect feedback
                feedback_choice = input("\nWould you like to provide feedback? (y/n): ").strip().lower()
                if feedback_choice == 'y':
                    rating = int(input("Rating (1-5): "))
                    comments = input("Comments (optional): ").strip()
                    correct_answer = None
                    if rating < 3:
                        correct_answer = input("Correct answer (if solution was wrong): ").strip()
                    
                    orchestrator.collect_feedback(
                        problem=problem,
                        category=result['routing']['category'],
                        solution=result['solution']['solution'],
                        rating=rating,
                        comments=comments,
                        correct_answer=correct_answer if correct_answer else None
                    )
                    print("✅ Thank you for your feedback!")
        
        elif choice == "2":
            stats = orchestrator.get_feedback_stats()
            print("\n📊 FEEDBACK STATISTICS:")
            print("=" * 60)
            print(f"Total Feedback: {stats['total_feedback']}")
            print(f"Average Rating: {stats['average_rating']:.2f}/5")
            print(f"Approved: {stats['approved_count']}")
            print(f"Approval Rate: {stats.get('approval_rate', 0)*100:.1f}%")
            
            if stats['category_stats']:
                print("\nCategory Performance:")
                for cat, cat_stats in stats['category_stats'].items():
                    print(f"\n{cat.upper()}:")
                    print(f"  Count: {cat_stats['count']}")
                    print(f"  Avg Rating: {cat_stats['average_rating']:.2f}/5")
                    print(f"  Approved: {cat_stats['approved']}")
        
        elif choice == "3":
            insights = orchestrator.get_learning_insights()
            print("\n💡 LEARNING INSIGHTS:")
            print("=" * 60)
            if not insights:
                print("No issues found. All solutions are performing well!")
            else:
                for insight in insights:
                    print(f"\n{insight['category'].upper()} - {insight['issue_count']} issues")
                    for idx, example in enumerate(insight['examples'], 1):
                        print(f"\nExample {idx}:")
                        print(f"  Problem: {example['problem']}")
                        if example.get('comments'):
                            print(f"  Feedback: {example['comments']}")
        
        elif choice == "4":
            print("\nAvailable Models:")
            print("Groq:", ", ".join(settings.GROQ_MODELS))
            print("Gemini:", ", ".join(settings.GEMINI_MODELS))
            
            provider = input("\nEnter provider (groq/gemini): ").strip().lower()
            model = input("Enter model name: ").strip()
            
            try:
                orchestrator.change_model(model, provider)
                print(f"✅ Model changed to {model}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        elif choice == "5":
            print("\nThank you for using Math Routing Agent! 👋")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
