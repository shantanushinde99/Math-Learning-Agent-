"""
DSPy Feedback Optimizer - Uses DSPy to optimize prompts based on user feedback
"""
import os
import json
from typing import List, Dict, Any, Optional
import dspy
from src.utils.logger import app_logger
from config.settings import settings


class MathSolverSignature(dspy.Signature):
    """Signature for mathematical problem solving with step-by-step reasoning"""
    problem = dspy.InputField(desc="The mathematical problem to solve")
    category = dspy.InputField(desc="The category of math problem (algebra, calculus, geometry, statistics)")
    solution = dspy.OutputField(desc="Detailed step-by-step solution with LaTeX formatting")


class DSPyFeedbackOptimizer:
    """
    Uses DSPy framework to optimize prompts and reasoning based on user feedback
    
    This provides the BONUS ADVANTAGE by:
    1. Learning from positive and negative feedback
    2. Optimizing prompts automatically using DSPy's optimization algorithms
    3. Improving solution quality over time
    4. Category-specific optimization
    """
    
    def __init__(self, provider: str = "groq", model: str = "llama-3.3-70b-versatile"):
        """
        Initialize DSPy optimizer
        
        Args:
            provider: LLM provider (groq or gemini)
            model: Model name to use
        """
        self.provider = provider
        self.model_name = model
        self.optimizer_dir = os.path.join(settings.DATA_DIR, "dspy_optimized")
        os.makedirs(self.optimizer_dir, exist_ok=True)
        
        # Configure DSPy with the appropriate LLM
        self._configure_dspy()
        
        # Create base solver
        self.base_solver = dspy.ChainOfThought(MathSolverSignature)
        
        # Load optimized solvers if they exist
        self.optimized_solvers = self._load_optimized_solvers()
        
        app_logger.info("[DSPY] DSPy Feedback Optimizer initialized")
    
    def _configure_dspy(self):
        """Configure DSPy with the appropriate LLM"""
        if self.provider == "groq":
            # Configure for Groq
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                lm = dspy.LM(
                    model=f"groq/{self.model_name}",
                    api_key=api_key,
                    temperature=0.1
                )
                dspy.configure(lm=lm)
                app_logger.info(f"[DSPY] Configured with Groq model: {self.model_name}")
            else:
                app_logger.warning("[DSPY] GROQ_API_KEY not found, using default configuration")
        
        elif self.provider == "gemini":
            # Configure for Gemini
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                lm = dspy.LM(
                    model=f"google/{self.model_name}",
                    api_key=api_key,
                    temperature=0.1
                )
                dspy.configure(lm=lm)
                app_logger.info(f"[DSPY] Configured with Gemini model: {self.model_name}")
            else:
                app_logger.warning("[DSPY] GOOGLE_API_KEY not found, using default configuration")
    
    def solve_with_dspy(self, problem: str, category: str) -> Dict[str, Any]:
        """
        Solve a problem using DSPy (optimized if available)
        
        Args:
            problem: The math problem
            category: Problem category
            
        Returns:
            Dict with solution and metadata
        """
        try:
            # Use optimized solver if available for this category
            solver = self.optimized_solvers.get(category, self.base_solver)
            
            # Generate solution
            result = solver(problem=problem, category=category)
            
            return {
                "solution": result.solution,
                "optimized": category in self.optimized_solvers,
                "category": category
            }
        
        except Exception as e:
            app_logger.error(f"[DSPY] Error solving with DSPy: {str(e)}")
            return {
                "solution": f"Error: {str(e)}",
                "optimized": False,
                "category": category
            }
    
    def optimize_from_feedback(
        self, 
        feedback_data: List[Dict[str, Any]], 
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize prompts based on user feedback using DSPy's optimization
        
        Args:
            feedback_data: List of feedback entries
            category: Specific category to optimize (None for all)
            
        Returns:
            Optimization results
        """
        if not feedback_data:
            app_logger.warning("[DSPY] No feedback data provided for optimization")
            return {"status": "no_data", "optimized_categories": []}
        
        # Group feedback by category
        category_feedback = self._group_feedback_by_category(feedback_data)
        
        # Filter by category if specified
        if category:
            category_feedback = {category: category_feedback.get(category, [])}
        
        optimized_categories = []
        
        for cat, feedback_list in category_feedback.items():
            if len(feedback_list) < 3:  # Need at least 3 examples
                app_logger.info(f"[DSPY] Skipping {cat}: insufficient feedback ({len(feedback_list)} examples)")
                continue
            
            try:
                # Create training examples from feedback
                trainset = self._create_training_examples(feedback_list)
                
                if len(trainset) < 2:
                    continue
                
                # Use BootstrapFewShot optimizer (works well for small datasets)
                optimizer = dspy.BootstrapFewShot(
                    metric=self._feedback_metric,
                    max_bootstrapped_demos=min(4, len(trainset)),
                    max_labeled_demos=min(8, len(trainset))
                )
                
                # Optimize the solver
                app_logger.info(f"[DSPY] Optimizing for category: {cat} with {len(trainset)} examples")
                optimized_solver = optimizer.compile(
                    self.base_solver,
                    trainset=trainset
                )
                
                # Save optimized solver
                self.optimized_solvers[cat] = optimized_solver
                self._save_optimized_solver(cat, optimized_solver)
                
                optimized_categories.append(cat)
                app_logger.info(f"[DSPY] ✅ Optimized solver for {cat}")
            
            except Exception as e:
                app_logger.error(f"[DSPY] Error optimizing {cat}: {str(e)}")
        
        return {
            "status": "success" if optimized_categories else "no_optimization",
            "optimized_categories": optimized_categories,
            "total_feedback": len(feedback_data)
        }
    
    def _group_feedback_by_category(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group feedback by category"""
        grouped = {}
        for entry in feedback_data:
            cat = entry.get("category", "general")
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(entry)
        return grouped
    
    def _create_training_examples(self, feedback_list: List[Dict[str, Any]]) -> List[dspy.Example]:
        """
        Create DSPy training examples from feedback
        
        Only uses highly-rated (4-5 stars) solutions as positive examples
        """
        examples = []
        
        for entry in feedback_list:
            # Only use approved feedback (rating >= 4) for training
            if entry.get("approved", False) and entry.get("rating", 0) >= 4:
                example = dspy.Example(
                    problem=entry["problem"],
                    category=entry["category"],
                    solution=entry["solution"]
                ).with_inputs("problem", "category")
                
                examples.append(example)
        
        return examples
    
    def _feedback_metric(self, example: dspy.Example, pred, trace=None) -> float:
        """
        Metric for evaluating solutions based on feedback quality
        
        Returns a score between 0.0 and 1.0
        """
        # Simple metric: solution should be detailed and contain steps
        solution = pred.solution if hasattr(pred, 'solution') else str(pred)
        
        score = 0.0
        
        # Check for step-by-step structure
        if "step" in solution.lower() or ":" in solution:
            score += 0.3
        
        # Check for LaTeX formatting
        if "$" in solution or "$$" in solution:
            score += 0.3
        
        # Check for explanation
        if len(solution) > 100:  # Detailed solution
            score += 0.2
        
        # Check for mathematical operators
        if any(op in solution for op in ["=", "+", "-", "*", "/"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _save_optimized_solver(self, category: str, solver):
        """Save optimized solver to disk"""
        try:
            filepath = os.path.join(self.optimizer_dir, f"{category}_optimized.json")
            solver.save(filepath)
            app_logger.info(f"[DSPY] Saved optimized solver for {category}")
        except Exception as e:
            app_logger.warning(f"[DSPY] Could not save solver: {str(e)}")
    
    def _load_optimized_solvers(self) -> Dict[str, Any]:
        """Load previously optimized solvers"""
        solvers = {}
        
        if not os.path.exists(self.optimizer_dir):
            return solvers
        
        try:
            for filename in os.listdir(self.optimizer_dir):
                if filename.endswith("_optimized.json"):
                    category = filename.replace("_optimized.json", "")
                    filepath = os.path.join(self.optimizer_dir, filename)
                    
                    # Try to load the solver
                    try:
                        solver = dspy.ChainOfThought(MathSolverSignature)
                        solver.load(filepath)
                        solvers[category] = solver
                        app_logger.info(f"[DSPY] Loaded optimized solver for {category}")
                    except Exception as e:
                        app_logger.warning(f"[DSPY] Could not load {category} solver: {str(e)}")
        
        except Exception as e:
            app_logger.warning(f"[DSPY] Error loading optimized solvers: {str(e)}")
        
        return solvers
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of optimized solvers"""
        return {
            "optimized_categories": list(self.optimized_solvers.keys()),
            "total_optimized": len(self.optimized_solvers),
            "provider": self.provider,
            "model": self.model_name
        }
