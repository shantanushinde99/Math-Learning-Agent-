"""
Orchestrator - Coordinates all agents in the math routing pipeline
WITH AI GATEWAY GUARDRAILS (Input + Output validation)
WITH DSPY FEEDBACK OPTIMIZATION (Bonus Feature)
"""
from typing import Dict, Any, Optional
from src.agents.router_agent import RouterAgent
from src.agents.solver_agent import SolverAgent
from src.agents.feedback_agent import FeedbackAgent
from src.tools.dspy_optimizer import DSPyFeedbackOptimizer
from src.utils.logger import app_logger
from src.utils.llm_factory import LLMFactory


class MathAgentOrchestrator:
    """
    Orchestrates the entire math problem solving pipeline with:
    - AI Gateway guardrails
    - DSPy feedback optimization (BONUS FEATURE)
    """
    
    def __init__(
        self, 
        model: Optional[str] = None, 
        provider: Optional[str] = None,
        enable_guardrails: bool = True,
        strict_mode: bool = True,
        enable_dspy: bool = True
    ):
        """
        Initialize orchestrator with AI Gateway guardrails and DSPy optimization
        
        Args:
            model: Model to use for agents
            provider: LLM provider to use
            enable_guardrails: Enable AI Gateway input/output guardrails
            strict_mode: Strict mathematics-only validation
            enable_dspy: Enable DSPy feedback optimization (BONUS)
        """
        self.llm = LLMFactory.create_llm(provider=provider, model=model) if model or provider else None
        self.enable_guardrails = enable_guardrails
        self.strict_mode = strict_mode
        self.enable_dspy = enable_dspy
        
        # Initialize agents with guardrails
        self.router = RouterAgent(
            llm=self.llm, 
            enable_guardrails=enable_guardrails,
            strict_mode=strict_mode
        )
        self.solver = SolverAgent(
            llm=self.llm,
            enable_guardrails=enable_guardrails
        )
        self.feedback_agent = FeedbackAgent()
        
        # Initialize DSPy optimizer (BONUS FEATURE)
        self.dspy_optimizer = None
        if enable_dspy:
            try:
                self.dspy_optimizer = DSPyFeedbackOptimizer(
                    provider=provider or "groq",
                    model=model or "llama-3.3-70b-versatile"
                )
                app_logger.info("[ORCHESTRATOR] ✨ DSPy Feedback Optimizer ENABLED")
            except Exception as e:
                app_logger.warning(f"[ORCHESTRATOR] DSPy initialization failed: {str(e)}")
                self.enable_dspy = False
        
        if enable_guardrails:
            app_logger.info(f"[ORCHESTRATOR] AI Gateway Guardrails ENABLED (strict_mode={strict_mode})")
        else:
            app_logger.info("[ORCHESTRATOR] AI Gateway Guardrails DISABLED")
        
    def process_problem(
        self,
        problem: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a math problem through the complete pipeline
        WITH AI GATEWAY GUARDRAILS (Input validation -> Routing -> Solving -> Output validation)
        
        Args:
            problem: The math problem to solve
            model: Optional specific model to use for solving
            
        Returns:
            Dict containing routing, solution, metadata, and guardrail status
        """
        app_logger.info("=" * 50)
        app_logger.info(f"Processing problem: {problem[:100]}...")
        
        # Step 1: Route the problem (includes INPUT GUARDRAIL)
        routing_result = self.router.route_problem(problem)
        
        # Check if input was blocked by guardrail
        if routing_result.get("guardrail_status") == "blocked":
            app_logger.warning("[ORCHESTRATOR] Problem BLOCKED by input guardrail")
            return {
                "problem": problem,
                "routing": routing_result,
                "solution": {
                    "solution": routing_result["reasoning"],
                    "category": "blocked",
                    "problem": problem
                },
                "pipeline_status": "blocked_by_guardrail",
                "guardrail_blocked": True
            }
        
        category = routing_result["category"]
        
        app_logger.info(f"Problem routed to: {category}")
        app_logger.info(f"Routing reasoning: {routing_result['reasoning']}")
        
        # Step 2: Solve the problem (includes OUTPUT GUARDRAIL)
        solution_result = self.solver.solve_problem(
            problem=problem,
            category=category,
            model=model
        )
        
        # Combine results
        result = {
            "problem": problem,
            "routing": routing_result,
            "solution": solution_result,
            "pipeline_status": "success",
            "guardrail_input_status": routing_result.get("guardrail_status", "disabled"),
            "guardrail_output_status": solution_result.get("guardrail_status", "disabled")
        }
        
        app_logger.info("Problem processing complete")
        app_logger.info("=" * 50)
        
        return result
    
    def collect_feedback(
        self,
        problem: str,
        category: str,
        solution: str,
        rating: int,
        comments: str = "",
        correct_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect feedback for a solution
        
        Args:
            problem: The math problem
            category: Problem category
            solution: The solution provided
            rating: Rating from 1-5
            comments: Feedback comments
            correct_answer: Correct answer if solution was wrong
            
        Returns:
            Feedback record
        """
        return self.feedback_agent.collect_feedback(
            problem=problem,
            category=category,
            solution=solution,
            rating=rating,
            comments=comments,
            correct_answer=correct_answer
        )
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        return self.feedback_agent.get_feedback_stats()
    
    def get_learning_insights(self):
        """Get learning insights from feedback"""
        return self.feedback_agent.get_learning_insights()
    
    def optimize_with_dspy(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize prompts using DSPy based on feedback 
        
        Args:
            category: Specific category to optimize (None for all)
            
        Returns:
            Optimization results
        """
        if not self.enable_dspy or not self.dspy_optimizer:
            return {
                "status": "disabled",
                "message": "DSPy optimization is not enabled"
            }
        
        # Get all feedback
        feedback_data = self.feedback_agent._load_feedback()
        
        if not feedback_data:
            return {
                "status": "no_data",
                "message": "No feedback data available for optimization"
            }
        
        # Run optimization
        app_logger.info("[ORCHESTRATOR] Starting DSPy optimization from feedback...")
        result = self.dspy_optimizer.optimize_from_feedback(feedback_data, category)
        
        app_logger.info(f"[ORCHESTRATOR] DSPy optimization complete: {result}")
        return result
    
    def solve_with_dspy(self, problem: str, category: str) -> Dict[str, Any]:
        """
        Solve problem using DSPy-optimized solver (if available)
        
        Args:
            problem: Math problem
            category: Problem category
            
        Returns:
            Solution result
        """
        if not self.enable_dspy or not self.dspy_optimizer:
            return {
                "error": "DSPy not enabled",
                "solution": None
            }
        
        return self.dspy_optimizer.solve_with_dspy(problem, category)
    
    def get_dspy_status(self) -> Dict[str, Any]:
        """Get DSPy optimization status"""
        if not self.enable_dspy or not self.dspy_optimizer:
            return {
                "enabled": False,
                "optimized_categories": []
            }
        
        status = self.dspy_optimizer.get_optimization_status()
        status["enabled"] = True
        return status
    
    def change_model(self, model: str, provider: Optional[str] = None):
        """
        Change the model used by agents
        
        Args:
            model: New model name
            provider: Optional new provider
        """
        app_logger.info(f"Changing model to: {model}")
        self.llm = LLMFactory.create_llm(provider=provider, model=model)
        
        # Reinitialize agents with new model and guardrails
        self.router = RouterAgent(
            llm=self.llm,
            enable_guardrails=self.enable_guardrails,
            strict_mode=self.strict_mode
        )
        self.solver = SolverAgent(
            llm=self.llm,
            enable_guardrails=self.enable_guardrails
        )
        
        # Reinitialize DSPy optimizer with new model
        if self.enable_dspy:
            try:
                self.dspy_optimizer = DSPyFeedbackOptimizer(
                    provider=provider or "groq",
                    model=model
                )
                app_logger.info("[ORCHESTRATOR] DSPy optimizer updated with new model")
            except Exception as e:
                app_logger.warning(f"[ORCHESTRATOR] DSPy reinitialization failed: {str(e)}")
