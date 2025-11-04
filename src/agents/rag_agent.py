"""
RAG Agent for Book-Based Learning
Uses vector store to retrieve relevant context and answer questions
WITH AI GATEWAY GUARDRAILS for mathematics education focus
"""
from typing import Dict, Any, List, Optional
from src.tools.vector_store import VectorStoreManager
from src.utils.llm_factory import LLMFactory
from src.utils.logger import app_logger
from config.settings import settings
from src.agents.guardrails import GuardrailManager


class RAGAgent:
    """Agent for Retrieval-Augmented Generation using uploaded documents"""
    
    def __init__(
        self, 
        model: str = None, 
        provider: str = None,
        enable_guardrails: bool = True,
        strict_mode: bool = True
    ):
        """
        Initialize RAG Agent with AI Gateway guardrails
        
        Args:
            model: LLM model name (optional)
            provider: LLM provider (optional)
            enable_guardrails: Enable AI Gateway input/output guardrails
            strict_mode: Strict mathematics-only validation
        """
        self.vector_store = VectorStoreManager()
        
        # Initialize LLM
        self.model = model or settings.LLM_MODEL
        self.provider = provider or settings.LLM_PROVIDER
        self.llm = LLMFactory.create_llm(self.provider, self.model)
        
        # Initialize guardrails
        self.enable_guardrails = enable_guardrails
        self.guardrail_manager = GuardrailManager(llm=self.llm, strict_mode=strict_mode) if enable_guardrails else None
        
        if enable_guardrails:
            app_logger.info(f"✅ RAG Agent initialized with {self.provider}/{self.model} + AI Gateway Guardrails (strict_mode={strict_mode})")
        else:
            app_logger.info(f"✅ RAG Agent initialized with {self.provider}/{self.model}")

    
    def answer_question(
        self,
        question: str,
        category: Optional[str] = None,
        n_context: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG
        WITH AI GATEWAY GUARDRAILS (Input validation + Output validation)
        
        Args:
            question: User's question
            category: Optional category filter
            n_context: Number of context chunks to retrieve
            
        Returns:
            Dictionary with answer, metadata, and guardrail status
        """
        try:
            app_logger.info(f"🔍 Processing RAG question: {question[:100]}...")
            
            # STEP 1: AI GATEWAY INPUT GUARDRAIL
            if self.enable_guardrails:
                app_logger.info("[RAG AGENT] Running input guardrail validation...")
                input_validation = self.guardrail_manager.validate_input(question)
                
                if not input_validation["valid"]:
                    app_logger.warning(f"[RAG AGENT] ❌ Input BLOCKED by guardrail: {input_validation['reason']}")
                    return {
                        "success": False,
                        "answer": input_validation["reason"],
                        "sources": [],
                        "context_used": 0,
                        "guardrail_status": "input_blocked",
                        "guardrail_details": input_validation
                    }
                
                app_logger.info("[RAG AGENT] ✅ Input approved by guardrail")
            
            # STEP 2: Retrieve relevant context
            context_chunks = self.vector_store.search(
                query=question,
                n_results=n_context,
                category=category
            )
            
            if not context_chunks:
                return {
                    "success": False,
                    "answer": "No relevant information found in the knowledge base. Please upload relevant documents first.",
                    "sources": [],
                    "context_used": 0,
                    "guardrail_status": "no_context"
                }
            
            # STEP 3: Format context
            context_text = self._format_context(context_chunks)
            
            # STEP 4: Create prompt
            prompt = self._create_rag_prompt(question, context_text)
            
            # STEP 5: Generate answer
            raw_answer = self.llm.invoke(prompt).content
            
            # STEP 6: AI GATEWAY OUTPUT GUARDRAIL
            final_answer = raw_answer
            guardrail_output_status = "approved"
            
            if self.enable_guardrails:
                app_logger.info("[RAG AGENT] Running output guardrail validation...")
                output_validation = self.guardrail_manager.validate_output(
                    raw_answer, 
                    question,
                    category
                )
                
                if not output_validation["approved"]:
                    app_logger.warning(f"[RAG AGENT] ⚠️ Output flagged by guardrail: {output_validation['issues']}")
                    final_answer = output_validation["modified_response"]
                    guardrail_output_status = "modified"
                else:
                    app_logger.info("[RAG AGENT] ✅ Output approved by guardrail")
            
            # STEP 7: Extract sources
            sources = self._extract_sources(context_chunks)
            
            result = {
                "success": True,
                "question": question,
                "answer": final_answer,
                "sources": sources,
                "context_used": len(context_chunks),
                "model_used": f"{self.provider}/{self.model}",
                "guardrail_status": guardrail_output_status
            }
            
            app_logger.info(f"✅ RAG answer generated using {len(context_chunks)} context chunks")
            
            return result
            
        except Exception as e:
            app_logger.error(f"❌ Error in RAG answer: {str(e)}")
            return {
                "success": False,
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "context_used": 0,
                "guardrail_status": "error"
            }
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks into a single string"""
        formatted = []
        
        for idx, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            text = chunk['text']
            formatted.append(f"[Source {idx}: {source}]\n{text}\n")
        
        return "\n".join(formatted)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source documents"""
        sources = set()
        for chunk in chunks:
            source = chunk['metadata'].get('source', 'Unknown')
            sources.add(source)
        return list(sources)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a RAG prompt with context and question"""
        return f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context
2. **IMPORTANT: Use LaTeX formatting for ALL mathematical expressions**
   - Wrap inline math in $...$ (e.g., $x = 5$)
   - Wrap display equations in $$...$$ (e.g., $$2x + 5 = 15$$)
3. Structure your answer with clear sections:
   - **Understanding**: Explain what the question is asking
   - **Step-by-step Explanation**: Show all work with proper LaTeX
   - **Final Answer**: Provide the conclusion clearly
4. Be precise and cite which source document(s) you're using
5. If the context doesn't contain enough information, say so clearly

LATEX FORMATTING GUIDE:
- Fractions: $$\\frac{{a}}{{b}}$$
- Square root: $$\\sqrt{{x}}$$
- Exponents: $$x^2, x^{{n+1}}$$
- Subscripts: $$x_1, x_{{i+1}}$$
- Derivatives: $$\\frac{{d}}{{dx}}f(x)$$
- Integrals: $$\\int x^2 dx$$
- Limits: $$\\lim_{{x \\to 0}} f(x)$$
- Summation: $$\\sum_{{i=1}}^{{n}} x_i$$
- Greek letters: $\\alpha, \\beta, \\theta, \\pi, \\mu, \\sigma$
- Relations: $$x \\in A, x \\notin B$$
- Logic: $$\\forall, \\exists, \\implies, \\iff$$
- Sets: $$\\{{1, 2, 3\\}}, A \\cup B, A \\cap B$$

ANSWER:"""
    
    def explain_concept(
        self,
        concept: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain a concept using the knowledge base
        
        Args:
            concept: Concept to explain
            category: Optional category filter
            
        Returns:
            Dictionary with explanation and sources
        """
        question = f"""Explain the concept of {concept} in detail with examples.
        
Use proper LaTeX formatting for all mathematical expressions, formulas, and equations.
Structure your explanation clearly with definitions, key formulas, and worked examples."""
        return self.answer_question(question, category, n_context=7)
    
    def find_examples(
        self,
        topic: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find examples related to a topic
        
        Args:
            topic: Topic to find examples for
            category: Optional category filter
            
        Returns:
            Dictionary with examples and sources
        """
        question = f"""Provide examples and practice problems related to {topic}.
        
For each example:
1. State the problem clearly using LaTeX for all math
2. Show step-by-step solution with proper LaTeX formatting
3. Highlight key concepts used"""
        return self.answer_question(question, category, n_context=5)
    
    def summarize_topic(
        self,
        topic: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize a topic from the knowledge base
        
        Args:
            topic: Topic to summarize
            category: Optional category filter
            
        Returns:
            Dictionary with summary and sources
        """
        question = f"""Provide a comprehensive summary of {topic}, including key points and formulas.

Structure the summary with:
1. **Overview**: Brief introduction
2. **Key Concepts**: Main ideas with LaTeX-formatted formulas
3. **Important Formulas**: All relevant equations in LaTeX
4. **Applications**: Where and how these concepts are used"""
        return self.answer_question(question, category, n_context=8)
    
    def change_model(self, model: str, provider: str = None):
        """Change the LLM model"""
        self.model = model
        if provider:
            self.provider = provider
        
        self.llm = LLMFactory.create_llm(self.provider, self.model)
        app_logger.info(f"🔄 RAG Agent model changed to {self.provider}/{self.model}")
