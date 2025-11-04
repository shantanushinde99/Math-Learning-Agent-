"""
Router Agent - Routes math problems to appropriate specialized agents
WITH AI GATEWAY GUARDRAILS for mathematics-focused content
"""
from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_factory import LLMFactory
from src.utils.logger import app_logger
from src.agents.guardrails import InputGuardrail


class RouterAgent:
    """Agent responsible for routing math problems to appropriate specialists"""
    
    PROBLEM_TYPES = ["algebra", "calculus", "geometry", "statistics", "general"]
    
    def __init__(self, llm=None, enable_guardrails: bool = True, strict_mode: bool = True):
        """
        Initialize router agent with AI Gateway guardrails
        
        Args:
            llm: Language model to use
            enable_guardrails: Enable input validation guardrails
            strict_mode: Strict mathematics-only validation
        """
        self.llm = llm or LLMFactory.create_llm(temperature=0.2)
        self.routing_prompt = self._create_routing_prompt()
        self.enable_guardrails = enable_guardrails
        self.guardrail = InputGuardrail(llm=self.llm, strict_mode=strict_mode) if enable_guardrails else None
        
        if enable_guardrails:
            app_logger.info("[ROUTER AGENT] AI Gateway Input Guardrails ENABLED")
        else:
            app_logger.info("[ROUTER AGENT] Input Guardrails DISABLED")
        
    def _create_routing_prompt(self) -> ChatPromptTemplate:
        """Create prompt for routing decisions"""
        template = """You are an expert mathematics problem classifier. Analyze the given math problem and classify it into one of these categories:

Categories:
- algebra: Problems involving equations, variables, polynomials, systems of equations
- calculus: Problems involving derivatives, integrals, limits, differential equations
- geometry: Problems involving shapes, angles, areas, volumes, coordinate geometry
- statistics: Problems involving probability, data analysis, distributions, hypothesis testing
- general: Basic arithmetic, word problems, or problems that don't fit other categories

Math Problem: {problem}

Analyze the problem carefully and respond with ONLY the category name (one word: algebra, calculus, geometry, statistics, or general).
Also provide a brief reasoning (1-2 sentences) for your classification.

Format your response as:
Category: <category_name>
Reasoning: <your reasoning>
"""
        return ChatPromptTemplate.from_template(template)
    
    def route_problem(self, problem: str) -> Dict[str, Any]:
        """
        Route a math problem to the appropriate category
        WITH INPUT GUARDRAIL VALIDATION
        
        Args:
            problem: The math problem to route
            
        Returns:
            Dict containing category, reasoning, confidence, and guardrail validation
        """
        app_logger.info(f"Routing problem: {problem[:100]}...")
        
        # STEP 1: AI GATEWAY INPUT GUARDRAIL
        guardrail_result = None
        if self.enable_guardrails:
            app_logger.info("[ROUTER AGENT] Running input guardrail validation...")
            guardrail_result = self.guardrail.validate(problem)
            
            if not guardrail_result["valid"]:
                app_logger.warning(f"[ROUTER AGENT] ❌ Input BLOCKED by guardrail: {guardrail_result['reason']}")
                return {
                    "category": "blocked",
                    "reasoning": guardrail_result["reason"],
                    "raw_response": "",
                    "guardrail_status": "blocked",
                    "guardrail_details": guardrail_result
                }
            
            app_logger.info(f"[ROUTER AGENT] ✅ Input APPROVED by guardrail - Category: {guardrail_result['category']}")
        
        # STEP 2: Route the problem (if guardrail passed)
        try:
            # Get routing decision from LLM
            chain = self.routing_prompt | self.llm
            response = chain.invoke({"problem": problem})
            
            # Parse response
            content = response.content
            lines = content.strip().split('\n')
            
            category = "general"
            reasoning = ""
            
            for line in lines:
                if line.startswith("Category:"):
                    category = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            # Validate category
            if category not in self.PROBLEM_TYPES:
                app_logger.warning(f"Invalid category '{category}', defaulting to 'general'")
                category = "general"
            
            result = {
                "category": category,
                "reasoning": reasoning,
                "raw_response": content,
                "guardrail_status": "approved" if self.enable_guardrails else "disabled",
                "guardrail_details": guardrail_result
            }
            
            app_logger.info(f"Routed to category: {category}")
            return result
            
        except Exception as e:
            app_logger.error(f"Error routing problem: {str(e)}")
            return {
                "category": "general",
                "reasoning": f"Error in routing: {str(e)}",
                "raw_response": "",
                "guardrail_status": "approved" if self.enable_guardrails else "disabled",
                "guardrail_details": guardrail_result
            }
