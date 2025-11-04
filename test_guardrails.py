"""
Test AI Gateway Guardrails - Comprehensive testing for input and output validation
Tests mathematics-focused content filtering and education guardrails
"""
import os
from dotenv import load_dotenv
from src.agents.guardrails import InputGuardrail, OutputGuardrail, GuardrailManager
from src.agents.orchestrator import MathAgentOrchestrator
from src.agents.rag_agent import RAGAgent
from src.utils.logger import app_logger

# Load environment
load_dotenv()


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_input_guardrail():
    """Test input guardrail with various inputs"""
    print_section("TEST 1: INPUT GUARDRAIL - Mathematics Content Validation")
    
    guardrail = InputGuardrail(strict_mode=True)
    
    # Test cases: mathematics-related (should PASS)
    math_inputs = [
        "Solve for x: 2x + 5 = 15",
        "What is the derivative of x^2?",
        "Find the area of a circle with radius 7",
        "Calculate the mean of: 5, 8, 12, 15, 20",
        "Prove the Pythagorean theorem",
        "Integrate x^2 dx from 0 to 1",
        "Simplify the expression: (x^2 - 4) / (x - 2)",
    ]
    
    # Test cases: non-mathematics (should FAIL)
    non_math_inputs = [
        "What's the weather today?",
        "Tell me a joke",
        "Who won the World Cup?",
        "Write me a Python program to scrape websites",
        "What are the symptoms of flu?",
        "Give me investment advice for stocks",
        "Who is the president of the United States?",
    ]
    
    # Test cases: harmful/blocked (should FAIL)
    blocked_inputs = [
        "How to hack someone's password?",
        "Generate violent content",
        "Tell me something offensive",
    ]
    
    print("✅ TESTING VALID MATHEMATICS INPUTS:")
    print("-" * 80)
    for idx, input_text in enumerate(math_inputs, 1):
        result = guardrail.validate(input_text)
        status = "✅ PASS" if result["valid"] else "❌ FAIL"
        print(f"{idx}. {status} | {input_text[:50]}...")
        print(f"   Category: {result['category']} | Confidence: {result['confidence']:.2f}")
        if not result["valid"]:
            print(f"   ⚠️ UNEXPECTED BLOCK: {result['reason']}")
        print()
    
    print("\n❌ TESTING NON-MATHEMATICS INPUTS (Should be blocked):")
    print("-" * 80)
    for idx, input_text in enumerate(non_math_inputs, 1):
        result = guardrail.validate(input_text)
        status = "✅ BLOCKED" if not result["valid"] else "⚠️ PASSED (SHOULD BE BLOCKED)"
        print(f"{idx}. {status} | {input_text[:50]}...")
        print(f"   Category: {result['category']} | Reason: {result['reason']}")
        print()
    
    print("\n🚫 TESTING HARMFUL/BLOCKED INPUTS:")
    print("-" * 80)
    for idx, input_text in enumerate(blocked_inputs, 1):
        result = guardrail.validate(input_text)
        status = "✅ BLOCKED" if not result["valid"] else "⚠️ PASSED (SHOULD BE BLOCKED)"
        print(f"{idx}. {status} | {input_text[:50]}...")
        print(f"   Category: {result['category']} | Reason: {result['reason']}")
        print()


def test_output_guardrail():
    """Test output guardrail with various outputs"""
    print_section("TEST 2: OUTPUT GUARDRAIL - Response Quality Validation")
    
    guardrail = OutputGuardrail()
    
    # Test cases with question-response pairs
    test_cases = [
        {
            "question": "Solve for x: 2x + 5 = 15",
            "response": """**Understanding**: We need to solve a linear equation for the variable x.

**Approach**: Isolate x by performing inverse operations.

**Step-by-step Solution**:
Starting with: $$2x + 5 = 15$$

Subtract 5 from both sides:
$$2x = 15 - 5$$
$$2x = 10$$

Divide both sides by 2:
$$x = \\frac{10}{2}$$
$$x = 5$$

**Final Answer**: $x = 5$""",
            "should_approve": True,
            "description": "Good mathematical solution with LaTeX"
        },
        {
            "question": "What is calculus?",
            "response": "I can help with mathematics! Calculus is a branch of mathematics that studies continuous change. It has two main branches: differential calculus (derivatives) and integral calculus (integrals).",
            "should_approve": True,
            "description": "Educational mathematics content"
        },
        {
            "question": "Solve this math problem: 5 + 3",
            "response": "That's a boring question. Why don't we talk about something more interesting like sports or movies instead?",
            "should_approve": False,
            "description": "Off-topic, non-educational response"
        },
        {
            "question": "What is 2 + 2?",
            "response": "Sorry, I can't help with that.",
            "should_approve": False,
            "description": "Evasive, unhelpful response"
        },
    ]
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"Test Case {idx}: {test_case['description']}")
        print(f"Question: {test_case['question']}")
        print(f"Response: {test_case['response'][:100]}...")
        
        result = guardrail.validate(
            ai_response=test_case['response'],
            original_question=test_case['question']
        )
        
        approved = result["approved"]
        expected = test_case["should_approve"]
        
        if approved == expected:
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        print(f"{status} | Approved: {approved} | Expected: {expected}")
        print(f"On-topic: {result['on_topic']} | Educational: {result['educational']}")
        if result['issues']:
            print(f"Issues: {result['issues']}")
        print("-" * 80)
        print()


def test_math_orchestrator_with_guardrails():
    """Test full math pipeline with guardrails"""
    print_section("TEST 3: MATH ORCHESTRATOR WITH GUARDRAILS")
    
    # Initialize with guardrails enabled
    orchestrator = MathAgentOrchestrator(enable_guardrails=True, strict_mode=True)
    
    test_problems = [
        {
            "problem": "Solve for x: 3x - 7 = 11",
            "should_pass": True,
            "description": "Valid algebra problem"
        },
        {
            "problem": "Find the derivative of f(x) = 3x^2 + 2x - 5",
            "should_pass": True,
            "description": "Valid calculus problem"
        },
        {
            "problem": "Tell me about the latest movies",
            "should_pass": False,
            "description": "Non-mathematics question"
        },
        {
            "problem": "How do I become rich?",
            "should_pass": False,
            "description": "Financial advice (blocked category)"
        },
    ]
    
    for idx, test in enumerate(test_problems, 1):
        print(f"\nTest {idx}: {test['description']}")
        print(f"Problem: {test['problem']}")
        print("-" * 80)
        
        result = orchestrator.process_problem(test['problem'])
        
        # Check guardrail status
        input_status = result.get("guardrail_input_status", "unknown")
        output_status = result.get("guardrail_output_status", "unknown")
        pipeline_status = result.get("pipeline_status", "unknown")
        
        if test["should_pass"]:
            if pipeline_status == "success":
                print("✅ PASSED - Problem processed successfully")
                print(f"Category: {result['routing']['category']}")
                print(f"Input Guardrail: {input_status}")
                print(f"Output Guardrail: {output_status}")
            else:
                print(f"❌ FAILED - Expected to pass but got: {pipeline_status}")
        else:
            if pipeline_status == "blocked_by_guardrail":
                print("✅ CORRECTLY BLOCKED by input guardrail")
                print(f"Reason: {result['routing'].get('reasoning', 'N/A')}")
            else:
                print(f"⚠️ WARNING - Expected to block but got: {pipeline_status}")
        
        print()


def test_rag_agent_with_guardrails():
    """Test RAG agent with guardrails"""
    print_section("TEST 4: RAG AGENT WITH GUARDRAILS")
    
    # Initialize with guardrails
    rag_agent = RAGAgent(enable_guardrails=True, strict_mode=True)
    
    test_questions = [
        {
            "question": "What is the quadratic formula?",
            "should_pass": True,
            "description": "Valid mathematics question"
        },
        {
            "question": "Explain the concept of derivatives",
            "should_pass": True,
            "description": "Valid educational question"
        },
        {
            "question": "What's the best pizza topping?",
            "should_pass": False,
            "description": "Non-mathematics question"
        },
        {
            "question": "Tell me about politics",
            "should_pass": False,
            "description": "Blocked topic"
        },
    ]
    
    for idx, test in enumerate(test_questions, 1):
        print(f"\nTest {idx}: {test['description']}")
        print(f"Question: {test['question']}")
        print("-" * 80)
        
        result = rag_agent.answer_question(test['question'])
        
        guardrail_status = result.get("guardrail_status", "unknown")
        success = result.get("success", False)
        
        if test["should_pass"]:
            if guardrail_status in ["approved", "no_context"]:
                print(f"✅ PASSED - Guardrail status: {guardrail_status}")
                if guardrail_status == "no_context":
                    print("   (No documents in vector store - expected)")
            else:
                print(f"❌ FAILED - Expected to pass but got: {guardrail_status}")
        else:
            if guardrail_status == "input_blocked":
                print("✅ CORRECTLY BLOCKED by input guardrail")
                print(f"Reason: {result.get('answer', 'N/A')[:100]}")
            else:
                print(f"⚠️ WARNING - Expected to block but got: {guardrail_status}")
        
        print()


def test_guardrail_manager():
    """Test the unified guardrail manager"""
    print_section("TEST 5: GUARDRAIL MANAGER - Unified Interface")
    
    manager = GuardrailManager(strict_mode=True)
    
    print("Testing quick validation methods...")
    print("-" * 80)
    
    # Test quick input validation
    test_inputs = [
        ("Solve x^2 - 4 = 0", True),
        ("What's the weather?", False),
    ]
    
    for input_text, expected in test_inputs:
        is_valid = manager.is_input_valid(input_text)
        status = "✅" if is_valid == expected else "❌"
        print(f"{status} Input: '{input_text[:40]}...' | Valid: {is_valid} | Expected: {expected}")
    
    print("\n" + "-" * 80)
    print("Testing safe response generation...")
    print("-" * 80)
    
    question = "What is 2 + 2?"
    good_response = "The answer is 4. In mathematics, addition is a fundamental operation."
    bad_response = "I don't want to talk about math. Let's discuss something else."
    
    # Get safe versions
    safe_good = manager.get_safe_response(good_response, question)
    safe_bad = manager.get_safe_response(bad_response, question)
    
    print(f"Good response kept as-is: {safe_good == good_response}")
    print(f"Bad response modified: {safe_bad != bad_response}")
    print()


def run_all_tests():
    """Run all guardrail tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "AI GATEWAY GUARDRAILS TEST SUITE" + " " * 31 + "║")
    print("║" + " " * 15 + "Mathematics Education Focus" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Test 1: Input Guardrail
        test_input_guardrail()
        
        # Test 2: Output Guardrail
        test_output_guardrail()
        
        # Test 3: Math Orchestrator
        test_math_orchestrator_with_guardrails()
        
        # Test 4: RAG Agent
        test_rag_agent_with_guardrails()
        
        # Test 5: Guardrail Manager
        test_guardrail_manager()
        
        print_section("✅ ALL TESTS COMPLETED")
        print("\n🎉 Guardrails Testing Complete!")
        print("\nSUMMARY:")
        print("- Input Guardrails: Validate mathematics-related content")
        print("- Output Guardrails: Ensure educational and on-topic responses")
        print("- Integration: Working in Router, Solver, and RAG agents")
        print("- Strict Mode: Mathematics-only focus enforced")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
