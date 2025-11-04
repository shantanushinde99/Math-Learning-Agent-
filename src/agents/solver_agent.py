"""
Solver Agent - Solves math problems based on their category
WITH AI GATEWAY OUTPUT GUARDRAILS for mathematics education
"""
from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_factory import LLMFactory
from src.utils.logger import app_logger
from config.settings import settings
from src.agents.guardrails import OutputGuardrail


class SolverAgent:
    """Agent responsible for solving math problems"""
    
    def __init__(self, llm=None, enable_guardrails: bool = True):
        """
        Initialize solver agent with AI Gateway output guardrails
        
        Args:
            llm: Language model to use
            enable_guardrails: Enable output validation guardrails
        """
        self.llm = llm or LLMFactory.create_llm()
        self.solver_prompts = self._create_solver_prompts()
        self.enable_guardrails = enable_guardrails
        self.guardrail = OutputGuardrail(llm=self.llm) if enable_guardrails else None
        
        if enable_guardrails:
            app_logger.info("[SOLVER AGENT] AI Gateway Output Guardrails ENABLED")
        else:
            app_logger.info("[SOLVER AGENT] Output Guardrails DISABLED")
        
    def _create_solver_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Create specialized prompts for different problem types"""
        
        algebra_template = """You are an expert in Algebra. Solve the following problem step by step.

Problem: {problem}

IMPORTANT: Use LaTeX formatting for ALL mathematical expressions. Wrap inline math in $ and display math in $$.

Provide a detailed solution with:
1. **Understanding**: Explain what the problem is asking
2. **Approach**: Describe the method you'll use
3. **Step-by-step Solution**: Show all work clearly using LaTeX
4. **Final Answer**: Provide the final answer clearly in LaTeX

LaTeX Examples:
- Inline: $x = 5$
- Equation: $$2x + 5 = 15$$
- Fraction: $$\\frac{{a}}{{b}}$$
- Square root: $$\\sqrt{{x}}$$
- Exponent: $$x^2$$
- Subscript: $$x_1$$

Be thorough and educational in your explanation with proper LaTeX formatting.
"""
        
        calculus_template = """You are an expert in Calculus. Solve the following problem step by step.

Problem: {problem}

IMPORTANT: Use LaTeX formatting for ALL mathematical expressions. Wrap inline math in $ and display math in $$.

Provide a detailed solution with:
1. **Understanding**: Identify what calculus concept is being used
2. **Approach**: Describe the rules/theorems you'll apply
3. **Step-by-step Solution**: Show all derivatives/integrals/limits clearly using LaTeX
4. **Final Answer**: Provide the final answer clearly in LaTeX

LaTeX Examples:
- Derivative: $$\\frac{{d}}{{dx}}f(x)$$
- Integral: $$\\int x^2 dx$$
- Limit: $$\\lim_{{x \\to 0}} f(x)$$
- Partial: $$\\frac{{\\partial f}}{{\\partial x}}$$
- Sum: $$\\sum_{{i=1}}^{{n}} x_i$$

Be rigorous and show all mathematical steps with proper LaTeX formatting.
"""
        
        geometry_template = """You are an expert in Geometry. Solve the following problem step by step.

Problem: {problem}

IMPORTANT: Use LaTeX formatting for ALL mathematical expressions. Wrap inline math in $ and display math in $$.

Provide a detailed solution with:
1. **Understanding**: Identify the geometric shapes and relationships
2. **Approach**: State the geometric principles/formulas you'll use
3. **Step-by-step Solution**: Show all calculations using LaTeX
4. **Final Answer**: Provide the final answer with units in LaTeX

LaTeX Examples:
- Area: $$A = \\pi r^2$$
- Angle: $$\\theta = 90°$$
- Distance: $$d = \\sqrt{{(x_2-x_1)^2 + (y_2-y_1)^2}}$$
- Greek letters: $\\alpha, \\beta, \\theta, \\pi$

Be clear about all geometric properties used with proper LaTeX formatting.
"""
        
        statistics_template = """You are an expert in Statistics and Probability. Solve the following problem step by step.

Problem: {problem}

IMPORTANT: Use LaTeX formatting for ALL mathematical expressions. Wrap inline math in $ and display math in $$.

Provide a detailed solution with:
1. **Understanding**: Identify the statistical concept
2. **Approach**: State the formulas/distributions you'll use
3. **Step-by-step Solution**: Show all calculations clearly using LaTeX
4. **Final Answer**: Provide the final answer with interpretation in LaTeX

LaTeX Examples:
- Mean: $$\\mu = \\frac{{1}}{{n}}\\sum_{{i=1}}^{{n}} x_i$$
- Standard deviation: $$\\sigma = \\sqrt{{\\frac{{1}}{{n}}\\sum_{{i=1}}^{{n}}(x_i - \\mu)^2}}$$
- Probability: $$P(A \\cap B) = P(A) \\cdot P(B|A)$$
- Normal distribution: $$N(\\mu, \\sigma^2)$$

Be precise with statistical notation and interpretation using proper LaTeX formatting.
"""
        
        general_template = """You are a mathematics expert. Solve the following problem step by step.

Problem: {problem}

IMPORTANT: Use LaTeX formatting for ALL mathematical expressions. Wrap inline math in $ and display math in $$.

Provide a detailed solution with:
1. **Understanding**: Explain what the problem is asking
2. **Approach**: Describe your solution method
3. **Step-by-step Solution**: Show all work clearly using LaTeX
4. **Final Answer**: Provide the final answer in LaTeX

LaTeX Examples:
- Basic: $x = 5$
- Equation: $$2x + 5 = 15$$
- Operations: $+, -, \\times, \\div$

Be clear and educational in your explanation with proper LaTeX formatting.
"""
        
        return {
            "algebra": ChatPromptTemplate.from_template(algebra_template),
            "calculus": ChatPromptTemplate.from_template(calculus_template),
            "geometry": ChatPromptTemplate.from_template(geometry_template),
            "statistics": ChatPromptTemplate.from_template(statistics_template),
            "general": ChatPromptTemplate.from_template(general_template)
        }
    
    def solve_problem(
        self,
        problem: str,
        category: str = "general",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a math problem using the appropriate specialized approach
        WITH OUTPUT GUARDRAIL VALIDATION
        
        Args:
            problem: The math problem to solve
            category: The category of the problem
            model: Optional model to use for solving
            
        Returns:
            Dict containing solution, metadata, and guardrail validation
        """
        app_logger.info(f"Solving {category} problem with model: {model or 'default'}")
        
        try:
            # Get temperature for this category
            temperature = settings.TEMPERATURE_CONFIG.get(category, 0.3)
            
            # Create LLM with appropriate temperature
            llm = LLMFactory.create_llm(
                model=model,
                temperature=temperature
            ) if model else self.llm
            
            # Get appropriate prompt
            prompt = self.solver_prompts.get(category, self.solver_prompts["general"])
            
            # STEP 1: Solve problem
            chain = prompt | llm
            response = chain.invoke({"problem": problem})
            raw_solution = response.content
            
            # STEP 2: AI GATEWAY OUTPUT GUARDRAIL
            guardrail_result = None
            final_solution = raw_solution
            
            if self.enable_guardrails:
                app_logger.info("[SOLVER AGENT] Running output guardrail validation...")
                guardrail_result = self.guardrail.validate(
                    ai_response=raw_solution,
                    original_question=problem,
                    category=category
                )
                
                if not guardrail_result["approved"]:
                    app_logger.warning(f"[SOLVER AGENT] ⚠️ Output flagged by guardrail: {guardrail_result['issues']}")
                    # Use the safe fallback response
                    final_solution = guardrail_result["modified_response"]
                else:
                    app_logger.info("[SOLVER AGENT] ✅ Output approved by guardrail")
            
            solution = {
                "problem": problem,
                "category": category,
                "solution": final_solution,
                "model_used": model or settings.LLM_MODEL,
                "temperature": temperature,
                "guardrail_status": "approved" if (not self.enable_guardrails or guardrail_result["approved"]) else "modified",
                "guardrail_details": guardrail_result
            }
            
            app_logger.info("Problem solved successfully")
            return solution
            
        except Exception as e:
            app_logger.error(f"Error solving problem: {str(e)}")
            return {
                "problem": problem,
                "category": category,
                "solution": f"Error solving problem: {str(e)}",
                "model_used": model or settings.LLM_MODEL,
                "temperature": 0.3,
                "error": str(e),
                "guardrail_status": "error",
                "guardrail_details": None
            }
