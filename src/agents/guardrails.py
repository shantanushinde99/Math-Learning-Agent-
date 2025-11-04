"""
AI Gateway Guardrails - Input and Output content validation for mathematics education
Ensures all interactions are focused on mathematics and educational content only
"""
from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_factory import LLMFactory
from src.utils.logger import app_logger


class InputGuardrail:
    """
    Input Guardrail - Validates incoming requests are mathematics-related and appropriate
    Acts as a gatekeeper before processing any user input
    """
    
    # Mathematics-related keywords for quick filtering
    MATH_KEYWORDS = [
        # General math terms
        "solve", "calculate", "find", "equation", "formula", "prove", "simplify",
        "evaluate", "compute", "determine", "derive", "integrate", "differentiate",
        # Math domains
        "algebra", "calculus", "geometry", "trigonometry", "statistics", "probability",
        "arithmetic", "matrix", "vector", "function", "polynomial", "logarithm",
        # Math operations
        "add", "subtract", "multiply", "divide", "sum", "product", "quotient",
        "derivative", "integral", "limit", "series", "sequence", "permutation",
        # Math objects
        "number", "angle", "triangle", "circle", "square", "rectangle", "sphere",
        "line", "point", "graph", "plot", "variable", "constant", "coefficient",
        # Advanced topics
        "theorem", "proof", "lemma", "corollary", "axiom", "postulate",
        "differential", "partial", "gradient", "divergence", "curl",
        # Symbols (when written out)
        "pi", "theta", "alpha", "beta", "gamma", "delta", "sigma", "infinity"
    ]
    
    # Blocked content types
    BLOCKED_CATEGORIES = [
        "harmful", "violent", "hateful", "sexual", "illegal", 
        "personal_info", "medical_advice", "financial_advice",
        "political", "religious_debate"
    ]
    
    def __init__(self, llm=None, strict_mode: bool = True):
        """
        Initialize input guardrail
        
        Args:
            llm: Language model for advanced validation
            strict_mode: If True, only allow mathematics content. If False, allow education-related content
        """
        self.llm = llm or LLMFactory.create_llm(temperature=0.0)  # Low temperature for consistency
        self.strict_mode = strict_mode
        self.validation_prompt = self._create_validation_prompt()
        
    def _create_validation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for input validation"""
        template = """You are an AI Gateway guardrail for a mathematics education platform. 
Your role is to determine if the user's input is appropriate and mathematics-related.

ALLOWED CONTENT:
- Mathematics problems (algebra, calculus, geometry, statistics, etc.)
- Educational questions about mathematical concepts
- Requests to explain math topics
- Math homework or exam questions
- Mathematical proofs and derivations

BLOCKED CONTENT:
- Non-educational content
- Harmful, violent, or hateful content
- Sexual or inappropriate content
- Personal information requests
- Medical, legal, or financial advice
- Off-topic questions (politics, entertainment, general knowledge)
- Code generation (unless for mathematical computation)
- Any content not related to mathematics education

User Input: {input_text}

Analyze the input and respond in this EXACT format:
VALID: [YES/NO]
CATEGORY: [mathematics/algebra/calculus/geometry/statistics/non-mathematics/blocked]
REASON: [Brief explanation]
CONFIDENCE: [0.0-1.0]

Be strict - only approve mathematics-related educational content."""
        return ChatPromptTemplate.from_template(template)
    
    def _quick_keyword_check(self, input_text: str) -> bool:
        """
        Quick keyword-based check for mathematics content
        Returns True if likely math-related, False otherwise
        """
        input_lower = input_text.lower()
        
        # Check for math keywords
        math_score = sum(1 for keyword in self.MATH_KEYWORDS if keyword in input_lower)
        
        # Check for mathematical symbols and patterns
        has_numbers = any(char.isdigit() for char in input_text)
        has_operators = any(op in input_text for op in ['+', '-', '*', '/', '=', '^', '√', '∫', '∑'])
        has_x_variable = ' x ' in input_lower or input_lower.startswith('x ') or input_lower.endswith(' x')
        
        # Calculate relevance score
        relevance_score = math_score
        if has_numbers:
            relevance_score += 1
        if has_operators:
            relevance_score += 2
        if has_x_variable:
            relevance_score += 1
            
        # Threshold: at least 2 points for likely math content
        return relevance_score >= 2
    
    def validate(self, input_text: str) -> Dict[str, Any]:
        """
        Validate input text through AI Gateway guardrail
        
        Args:
            input_text: User input to validate
            
        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "category": str,
                "reason": str,
                "confidence": float,
                "quick_check_passed": bool
            }
        """
        app_logger.info(f"[INPUT GUARDRAIL] Validating input: {input_text[:100]}...")
        
        # Quick check first (faster)
        quick_check = self._quick_keyword_check(input_text)
        
        if not quick_check and self.strict_mode:
            app_logger.warning("[INPUT GUARDRAIL] Quick check failed - likely not mathematics content")
            return {
                "valid": False,
                "category": "non-mathematics",
                "reason": "Input does not appear to be mathematics-related. Please ask mathematics questions only.",
                "confidence": 0.7,
                "quick_check_passed": False
            }
        
        # Advanced LLM-based validation
        try:
            chain = self.validation_prompt | self.llm
            response = chain.invoke({"input_text": input_text})
            content = response.content.strip()
            
            # Parse response
            valid = False
            category = "unknown"
            reason = ""
            confidence = 0.0
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith("VALID:"):
                    valid = "YES" in line.upper()
                elif line.startswith("CATEGORY:"):
                    category = line.split(":", 1)[1].strip().lower()
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
                        confidence = 0.5
            
            result = {
                "valid": valid,
                "category": category,
                "reason": reason,
                "confidence": confidence,
                "quick_check_passed": quick_check
            }
            
            if valid:
                app_logger.info(f"[INPUT GUARDRAIL] ✅ Input approved - Category: {category}")
            else:
                app_logger.warning(f"[INPUT GUARDRAIL] ❌ Input blocked - Reason: {reason}")
            
            return result
            
        except Exception as e:
            app_logger.error(f"[INPUT GUARDRAIL] Error in validation: {str(e)}")
            # Fail safe - if strict mode, block on error
            return {
                "valid": not self.strict_mode,
                "category": "error",
                "reason": f"Validation error: {str(e)}",
                "confidence": 0.0,
                "quick_check_passed": quick_check
            }


class OutputGuardrail:
    """
    Output Guardrail - Validates generated responses are educational and mathematics-focused
    Ensures AI responses stay on-topic and appropriate
    """
    
    def __init__(self, llm=None):
        """
        Initialize output guardrail
        
        Args:
            llm: Language model for validation
        """
        self.llm = llm or LLMFactory.create_llm(temperature=0.0)
        self.validation_prompt = self._create_validation_prompt()
        
    def _create_validation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for output validation"""
        template = """You are an AI Gateway output guardrail for a mathematics education platform.
Your role is to verify that the AI's response is appropriate and focused on mathematics education.

ACCEPTABLE RESPONSES:
- Mathematical solutions and explanations
- Step-by-step problem solving
- Educational content about math concepts
- Formulas, theorems, and proofs
- Examples and practice problems
- Helpful guidance for learning mathematics

UNACCEPTABLE RESPONSES:
- Off-topic content (non-mathematics)
- Harmful or inappropriate content
- Personal opinions on non-math topics
- Non-educational content
- Code without mathematical context
- Evasive or unhelpful responses

Original Question: {original_question}
AI Response: {ai_response}

Analyze the response and answer in this EXACT format:
APPROVED: [YES/NO]
ON_TOPIC: [YES/NO]
EDUCATIONAL: [YES/NO]
ISSUES: [List any issues, or "None"]
CONFIDENCE: [0.0-1.0]

Be strict - only approve educational mathematics responses."""
        return ChatPromptTemplate.from_template(template)
    
    def _quick_response_check(self, response: str, original_question: str) -> bool:
        """
        Quick sanity check on the response
        Returns True if response looks acceptable, False if clearly problematic
        """
        response_lower = response.lower()
        
        # Check for obviously problematic content
        blocked_phrases = [
            "i cannot help", "i can't assist with that", "sorry, i can't",
            "that's not appropriate", "i'm not able to",
            "as an ai", "i don't have opinions"
        ]
        
        # Check length - too short or too long might be problematic
        if len(response) < 20:
            return False
            
        # Check if response seems to address the question
        # Extract key terms from question
        question_words = set(original_question.lower().split())
        response_words = set(response_lower.split())
        
        # Some overlap expected
        overlap = len(question_words.intersection(response_words))
        
        return overlap > 0
    
    def validate(
        self, 
        ai_response: str, 
        original_question: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate AI-generated response through output guardrail
        
        Args:
            ai_response: The AI-generated response to validate
            original_question: The original user question
            category: Optional category of the question
            
        Returns:
            Dict with validation results:
            {
                "approved": bool,
                "on_topic": bool,
                "educational": bool,
                "issues": List[str],
                "confidence": float,
                "modified_response": str (if needed)
            }
        """
        app_logger.info(f"[OUTPUT GUARDRAIL] Validating response for: {original_question[:100]}...")
        
        # Quick check first
        quick_check = self._quick_response_check(ai_response, original_question)
        
        if not quick_check:
            app_logger.warning("[OUTPUT GUARDRAIL] Quick check failed - response may be problematic")
        
        # Advanced LLM-based validation
        try:
            chain = self.validation_prompt | self.llm
            response = chain.invoke({
                "original_question": original_question,
                "ai_response": ai_response
            })
            content = response.content.strip()
            
            # Parse response
            approved = False
            on_topic = False
            educational = False
            issues = []
            confidence = 0.0
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith("APPROVED:"):
                    approved = "YES" in line.upper()
                elif line.startswith("ON_TOPIC:"):
                    on_topic = "YES" in line.upper()
                elif line.startswith("EDUCATIONAL:"):
                    educational = "YES" in line.upper()
                elif line.startswith("ISSUES:"):
                    issues_text = line.split(":", 1)[1].strip()
                    if issues_text.lower() != "none":
                        issues = [i.strip() for i in issues_text.split(",")]
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
                        confidence = 0.5
            
            result = {
                "approved": approved,
                "on_topic": on_topic,
                "educational": educational,
                "issues": issues,
                "confidence": confidence,
                "modified_response": ai_response  # Return original if approved
            }
            
            if approved:
                app_logger.info("[OUTPUT GUARDRAIL] ✅ Response approved")
            else:
                app_logger.warning(f"[OUTPUT GUARDRAIL] ❌ Response blocked - Issues: {issues}")
                # Replace with safe fallback
                result["modified_response"] = self._create_fallback_response(original_question, category)
            
            return result
            
        except Exception as e:
            app_logger.error(f"[OUTPUT GUARDRAIL] Error in validation: {str(e)}")
            # On error, allow response but flag it
            return {
                "approved": True,
                "on_topic": True,
                "educational": True,
                "issues": [f"Validation error: {str(e)}"],
                "confidence": 0.0,
                "modified_response": ai_response
            }
    
    def _create_fallback_response(self, question: str, category: Optional[str] = None) -> str:
        """
        Create a safe fallback response when output validation fails
        
        Args:
            question: The original question
            category: Optional category
            
        Returns:
            Safe fallback response
        """
        category_text = f" related to {category}" if category else ""
        return f"""I apologize, but I can only provide help with mathematics education{category_text}. 

Your question was: "{question}"

Please rephrase your question to focus on mathematics concepts, problems, or educational topics. 

For example:
- "How do I solve this equation: 2x + 5 = 15?"
- "Explain the Pythagorean theorem"
- "What is the derivative of x^2?"

I'm here to help with your mathematics learning!"""


class GuardrailManager:
    """
    Manages both input and output guardrails
    Provides unified interface for AI Gateway
    """
    
    def __init__(self, llm=None, strict_mode: bool = True):
        """
        Initialize guardrail manager
        
        Args:
            llm: Language model for guardrails
            strict_mode: Strict mathematics-only validation
        """
        self.input_guardrail = InputGuardrail(llm=llm, strict_mode=strict_mode)
        self.output_guardrail = OutputGuardrail(llm=llm)
        self.strict_mode = strict_mode
        app_logger.info(f"[GUARDRAIL MANAGER] Initialized with strict_mode={strict_mode}")
    
    def validate_input(self, input_text: str) -> Dict[str, Any]:
        """Validate user input"""
        return self.input_guardrail.validate(input_text)
    
    def validate_output(
        self, 
        ai_response: str, 
        original_question: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate AI output"""
        return self.output_guardrail.validate(ai_response, original_question, category)
    
    def is_input_valid(self, input_text: str) -> bool:
        """Quick check if input is valid"""
        result = self.validate_input(input_text)
        return result["valid"]
    
    def is_output_approved(self, ai_response: str, original_question: str) -> bool:
        """Quick check if output is approved"""
        result = self.validate_output(ai_response, original_question)
        return result["approved"]
    
    def get_safe_response(
        self, 
        ai_response: str, 
        original_question: str,
        category: Optional[str] = None
    ) -> str:
        """
        Get a safe response - validates and returns approved or modified version
        
        Args:
            ai_response: AI-generated response
            original_question: Original question
            category: Optional category
            
        Returns:
            Safe response (original if approved, modified if not)
        """
        result = self.validate_output(ai_response, original_question, category)
        return result["modified_response"]
