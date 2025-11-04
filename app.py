"""
Streamlit App - Math Routing Agent with Human-in-the-Loop Feedback + Book-Based Learning (RAG)
"""
import streamlit as st
import os
from datetime import datetime
from src.agents.orchestrator import MathAgentOrchestrator
from src.agents.rag_agent import RAGAgent
from src.tools.history_manager import HistoryManager
from src.tools.vector_store import VectorStoreManager
from config.settings import settings
from src.utils.logger import app_logger

# Page configuration
st.set_page_config(
    page_title="AI Math & Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state - Math Agent
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MathAgentOrchestrator()
    
if "history_manager" not in st.session_state:
    st.session_state.history_manager = HistoryManager()
    
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# Initialize session state - RAG Agent
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = RAGAgent()
    
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Sidebar - Model Selection
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Provider selection
provider = st.sidebar.selectbox(
    "Select LLM Provider",
    options=["groq", "gemini"],
    index=0 if settings.LLM_PROVIDER == "groq" else 1
)

# Model selection based on provider
if provider == "groq":
    available_models = settings.GROQ_MODELS
else:
    available_models = settings.GEMINI_MODELS

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=available_models,
    index=0
)

# Apply model change
if st.sidebar.button("Apply Model Change"):
    st.session_state.orchestrator.change_model(selected_model, provider)
    st.sidebar.success(f"✅ Model changed to {selected_model}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Model:** {selected_model}")
st.sidebar.markdown(f"**Provider:** {provider.upper()}")

# Main App
st.title("🎓 AI Math & Learning Assistant")
st.markdown("### Intelligent Math Problem Solver + Book-Based Learning (RAG)")

# Main Section Selection
st.markdown("---")
section = st.radio(
    "Choose Learning Mode:",
    ["🧮 Math Problem Solver", "📚 Book-Based Learning (RAG)"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# ============================================================================
# SECTION 1: MATH PROBLEM SOLVER
# ============================================================================
if section == "🧮 Math Problem Solver":
    st.header("🧮 Math Problem Solver")
    st.markdown("*Intelligent routing to specialized math models*")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Solve Problem", "📊 Feedback Stats", "📚 History", "💡 Learning Insights"])

    # Tab 1: Solve Problem
    with tab1:
        st.markdown("#### Enter your math problem:")
        
        # Example problems
        examples = {
            "Algebra": "Solve for x: 2x + 5 = 15",
            "Calculus": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
            "Geometry": "Find the area of a circle with radius 7 cm",
            "Statistics": "Calculate the mean and standard deviation of: 5, 8, 12, 15, 20",
            "General": "If John has 15 apples and gives 6 to Mary, how many does he have left?"
        }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        problem_input = st.text_area(
            "Math Problem",
            height=100,
            placeholder="Enter your math problem here...",
            key="math_problem_input"
        )
    
    with col2:
        st.markdown("**Quick Examples:**")
        for category, example in examples.items():
            if st.button(f"📌 {category}", key=f"example_{category}"):
                problem_input = example
                st.rerun()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        solve_button = st.button("🚀 Solve Problem", type="primary", use_container_width=True)
    
    with col2:
        use_specific_model = st.checkbox("Use specific model")
    
    with col3:
        if use_specific_model:
            specific_model = st.selectbox(
                "Model for solving",
                options=available_models,
                label_visibility="collapsed"
            )
        else:
            specific_model = None
    
    if solve_button and problem_input:
        with st.spinner("🤔 Processing your problem..."):
            try:
                result = st.session_state.orchestrator.process_problem(
                    problem=problem_input,
                    model=specific_model
                )
                st.session_state.current_result = result
                st.session_state.history_manager.add_to_history(result)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display result
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("### 🎯 Results")
        
        # Routing Information
        st.markdown("#### 🧭 Problem Classification")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Category", result["routing"]["category"].upper())
        with col2:
            st.info(f"**Reasoning:** {result['routing']['reasoning']}")
        
        # Solution
        st.markdown("#### ✅ Solution")
        # Display solution with LaTeX rendering
        st.markdown(result["solution"]["solution"], unsafe_allow_html=False)
        
        st.markdown("---")
        
        # Feedback Section
        st.markdown("### 💬 Provide Feedback")
        st.markdown("Help us improve by rating this solution!")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            rating = st.slider(
                "Rating (1-5)",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = Poor, 5 = Excellent"
            )
        
        with col2:
            comments = st.text_area(
                "Comments (optional)",
                height=100,
                placeholder="Any suggestions or corrections?"
            )
        
        if rating < 3:
            correct_answer = st.text_area(
                "If incorrect, provide the correct answer:",
                height=80
            )
        else:
            correct_answer = None
        
        if st.button("📤 Submit Feedback", type="primary"):
            feedback = st.session_state.orchestrator.collect_feedback(
                problem=result["problem"],
                category=result["routing"]["category"],
                solution=result["solution"]["solution"],
                rating=rating,
                comments=comments,
                correct_answer=correct_answer
            )
            st.success("✅ Thank you for your feedback!")
            st.balloons()

    # Tab 2: Feedback Stats
    with tab2:
        st.markdown("### 📊 Feedback Statistics")
        
        stats = st.session_state.orchestrator.get_feedback_stats()
        
        if stats["total_feedback"] == 0:
            st.info("No feedback collected yet. Solve some problems and provide feedback!")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Feedback", stats["total_feedback"])
            with col2:
                st.metric("Avg Rating", f"{stats['average_rating']:.2f}/5")
            with col3:
                st.metric("Approved", stats["approved_count"])
            with col4:
                st.metric("Approval Rate", f"{stats['approval_rate']*100:.1f}%")
            
            st.markdown("---")
            st.markdown("#### 📈 Category Performance")
            
            if stats["category_stats"]:
                for category, cat_stats in stats["category_stats"].items():
                    with st.expander(f"📁 {category.upper()}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Problems", cat_stats["count"])
                        with col2:
                            st.metric("Avg Rating", f"{cat_stats['average_rating']:.2f}/5")
                        with col3:
                            st.metric("Approved", cat_stats["approved"])

    # Tab 3: History
    with tab3:
        st.markdown("### 📚 Problem History")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search history", placeholder="Enter keyword...")
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.history_manager.clear_history()
                st.success("History cleared!")
                st.rerun()
        
        # Get history
        if search_term:
            history = st.session_state.history_manager.search_history(search_term)
        else:
            history = st.session_state.history_manager.get_history(limit=20)
        
        if not history:
            st.info("No history found.")
        else:
            for idx, entry in enumerate(history):
                with st.expander(f"**{entry['category'].upper()}** - {entry['problem'][:80]}... ({entry['timestamp'][:10]})"):
                    st.markdown(f"**Problem:** {entry['problem']}")
                    st.markdown(f"**Category:** {entry['category']}")
                    st.markdown(f"**Model:** {entry['model_used']}")
                    st.markdown("**Solution:**")
                    # Display with LaTeX rendering
                    st.markdown(entry['solution'], unsafe_allow_html=False)

    # Tab 4: Learning Insights
    with tab4:
        st.markdown("### 💡 Learning Insights")
        st.markdown("Insights based on feedback to improve the system")
        
        insights = st.session_state.orchestrator.get_learning_insights()
        
        if not insights:
            st.info("No low-rated feedback yet. This section will show areas for improvement.")
        else:
            for insight in insights:
                with st.expander(f"⚠️ {insight['category'].upper()} - {insight['issue_count']} issues"):
                    st.markdown(f"**Problems to address:** {insight['issue_count']}")
                    
                    for idx, example in enumerate(insight['examples'], 1):
                        st.markdown(f"**Example {idx}:**")
                        st.markdown(f"- Problem: {example['problem']}")
                        if example.get('comments'):
                            st.markdown(f"- Feedback: {example['comments']}")
                        if example.get('correct_answer'):
                            st.markdown(f"- Correct Answer: {example['correct_answer']}")
                        st.markdown("---")

# ============================================================================
# SECTION 2: BOOK-BASED LEARNING (RAG)
# ============================================================================
elif section == "📚 Book-Based Learning (RAG)":
    st.header("📚 Book-Based Learning")
    st.markdown("*Upload documents and learn through Q&A with RAG (Retrieval-Augmented Generation)*")
    
    # Tabs for RAG section
    rag_tab1, rag_tab2, rag_tab3, rag_tab4 = st.tabs([
        "📤 Upload Documents", 
        "❓ Ask Questions", 
        "🔍 Explore Topics",
        "📜 Q&A History"
    ])
    
    # RAG Tab 1: Upload Documents
    with rag_tab1:
        st.markdown("### 📤 Upload Your Learning Materials")
        st.markdown("Upload PDF, TXT, DOCX, or Markdown files to build your knowledge base.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True,
                help="Supported formats: PDF, TXT, DOCX, MD"
            )
            
            category_input = st.text_input(
                "Category (optional)",
                placeholder="e.g., Mathematics, Physics, Programming...",
                help="Add a category to organize your documents"
            )
        
        with col2:
            st.markdown("#### 📊 Collection Stats")
            stats = st.session_state.rag_agent.vector_store.get_collection_stats()
            st.metric("Total Documents", stats.get("unique_documents", 0))
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            
            if stats.get("categories"):
                st.markdown("**Categories:**")
                for cat in stats["categories"]:
                    st.markdown(f"- {cat}")
        
        if uploaded_files:
            if st.button("📥 Upload & Process", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_files = len(uploaded_files)
                success_count = 0
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Save file temporarily
                        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
                        file_path = os.path.join(settings.UPLOAD_DIR, uploaded_file.name)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Add to vector store
                        st.session_state.rag_agent.vector_store.add_documents(
                            file_path,
                            category=category_input if category_input else None
                        )
                        
                        success_count += 1
                        progress_bar.progress((idx + 1) / total_files)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                status_text.empty()
                progress_bar.empty()
                
                if success_count > 0:
                    st.success(f"✅ Successfully uploaded {success_count}/{total_files} documents!")
                    st.balloons()
                    st.rerun()
        
        # Document Management
        st.markdown("---")
        st.markdown("### 🗂️ Manage Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                if st.session_state.rag_agent.vector_store.clear_collection():
                    st.success("All documents cleared!")
                    st.rerun()
        
        with col2:
            st.info("📝 Tip: Upload textbooks, notes, or reference materials to build your personalized knowledge base!")
    
    # RAG Tab 2: Ask Questions
    with rag_tab2:
        st.markdown("### ❓ Ask Questions About Your Documents")
        
        # Check if documents exist
        stats = st.session_state.rag_agent.vector_store.get_collection_stats()
        
        if stats.get("unique_documents", 0) == 0:
            st.warning("⚠️ No documents uploaded yet! Please upload documents in the 'Upload Documents' tab first.")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                question = st.text_area(
                    "Your Question",
                    height=100,
                    placeholder="Ask anything about your uploaded documents...",
                    key="rag_question_input"
                )
            
            with col2:
                st.markdown("**Filter by Category:**")
                categories = ["All"] + stats.get("categories", [])
                selected_category = st.selectbox(
                    "Category",
                    options=categories,
                    label_visibility="collapsed"
                )
                
                rag_model_selection = st.selectbox(
                    "Model",
                    options=available_models,
                    help="Choose the model for answering"
                )
            
            if st.button("🔍 Get Answer", type="primary"):
                if question:
                    with st.spinner("🤔 Searching documents and generating answer..."):
                        try:
                            # Change model if different
                            if rag_model_selection != selected_model:
                                st.session_state.rag_agent.change_model(rag_model_selection, provider)
                            
                            # Get answer
                            category_filter = None if selected_category == "All" else selected_category
                            result = st.session_state.rag_agent.answer_question(
                                question,
                                category=category_filter
                            )
                            
                            # Add to history
                            st.session_state.qa_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "question": question,
                                "answer": result["answer"],
                                "sources": result["sources"],
                                "category": selected_category
                            })
                            
                            # Display answer
                            st.markdown("---")
                            st.markdown("### ✅ Answer")
                            st.markdown(result["answer"])
                            
                            if result["sources"]:
                                with st.expander("📚 Sources"):
                                    for idx, source in enumerate(result["sources"], 1):
                                        st.markdown(f"**Source {idx}:**")
                                        st.markdown(source)
                                        st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("Please enter a question!")
    
    # RAG Tab 3: Explore Topics
    with rag_tab3:
        st.markdown("### 🔍 Explore Topics")
        st.markdown("Deep dive into specific concepts from your documents")
        
        stats = st.session_state.rag_agent.vector_store.get_collection_stats()
        
        if stats.get("unique_documents", 0) == 0:
            st.warning("⚠️ No documents uploaded yet!")
        else:
            exploration_mode = st.selectbox(
                "Exploration Mode",
                ["💡 Explain Concept", "📋 Find Examples", "📝 Summarize Topic"]
            )
            
            topic_input = st.text_input(
                "Enter topic or concept",
                placeholder="e.g., Pythagorean theorem, linear regression, etc."
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                categories = ["All"] + stats.get("categories", [])
                topic_category = st.selectbox(
                    "Category Filter",
                    options=categories
                )
            
            if st.button("🚀 Explore", type="primary"):
                if topic_input:
                    with st.spinner("🔍 Exploring your documents..."):
                        try:
                            category_filter = None if topic_category == "All" else topic_category
                            
                            if exploration_mode == "💡 Explain Concept":
                                result = st.session_state.rag_agent.explain_concept(
                                    topic_input,
                                    category=category_filter
                                )
                            elif exploration_mode == "📋 Find Examples":
                                result = st.session_state.rag_agent.find_examples(
                                    topic_input,
                                    category=category_filter
                                )
                            else:  # Summarize Topic
                                result = st.session_state.rag_agent.summarize_topic(
                                    topic_input,
                                    category=category_filter
                                )
                            
                            st.markdown("---")
                            st.markdown(f"### {exploration_mode}")
                            st.markdown(result["answer"])
                            
                            if result["sources"]:
                                with st.expander("📚 Sources"):
                                    for idx, source in enumerate(result["sources"], 1):
                                        st.markdown(f"**Source {idx}:**")
                                        st.markdown(source)
                                        st.markdown("---")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("Please enter a topic!")
    
    # RAG Tab 4: Q&A History
    with rag_tab4:
        st.markdown("### 📜 Question & Answer History")
        
        if not st.session_state.qa_history:
            st.info("No Q&A history yet. Start asking questions!")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.qa_history = []
                    st.success("History cleared!")
                    st.rerun()
            
            st.markdown(f"**Total Questions Asked:** {len(st.session_state.qa_history)}")
            st.markdown("---")
            
            for idx, entry in enumerate(reversed(st.session_state.qa_history)):
                with st.expander(f"**Q{len(st.session_state.qa_history) - idx}:** {entry['question'][:100]}... ({entry['timestamp']})"):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Category:** {entry.get('category', 'All')}")
                    st.markdown(f"**Timestamp:** {entry['timestamp']}")
                    st.markdown("**Answer:**")
                    st.markdown(entry['answer'])
                    
                    if entry.get('sources'):
                        with st.expander("📚 View Sources"):
                            for source_idx, source in enumerate(entry['sources'], 1):
                                st.markdown(f"**Source {source_idx}:**")
                                st.markdown(source)
                                st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎓 AI Math & Learning Assistant | Math Routing Agent + Book-Based Learning (RAG) | Powered by Groq & Gemini</p>
    </div>
    """,
    unsafe_allow_html=True
)
