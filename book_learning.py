"""
Book-Based Learning - Streamlit App
Upload documents and ask questions using RAG
"""
import streamlit as st
import os
from pathlib import Path
from src.agents.rag_agent import RAGAgent
from src.tools.vector_store import VectorStoreManager
from config.settings import settings
from src.utils.logger import app_logger

# Page configuration
st.set_page_config(
    page_title="Book-Based Learning - RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = RAGAgent()
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

if "rag_history" not in st.session_state:
    st.session_state.rag_history = []

# Sidebar - Settings
st.sidebar.title("📚 Book-Based Learning")
st.sidebar.markdown("---")

# Model Selection
st.sidebar.markdown("### ⚙️ Settings")

provider = st.sidebar.selectbox(
    "Select LLM Provider",
    options=["groq", "gemini"],
    index=0 if settings.LLM_PROVIDER == "groq" else 1,
    key="rag_provider"
)

if provider == "groq":
    available_models = settings.GROQ_MODELS
else:
    available_models = settings.GEMINI_MODELS

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=available_models,
    index=0,
    key="rag_model"
)

if st.sidebar.button("Apply Model Change", key="rag_apply_model"):
    st.session_state.rag_agent.change_model(selected_model, provider)
    st.sidebar.success(f"✅ Model changed to {selected_model}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Model:** {selected_model}")
st.sidebar.markdown(f"**Provider:** {provider.upper()}")

# Collection Stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Knowledge Base Stats")

stats = st.session_state.vector_store.get_collection_stats()

st.sidebar.metric("Total Chunks", stats["total_chunks"])
st.sidebar.metric("Unique Documents", stats["unique_documents"])

if stats["document_names"]:
    with st.sidebar.expander("📄 Uploaded Documents"):
        for doc_name in stats["document_names"]:
            st.sidebar.text(f"• {doc_name}")

# Main App
st.title("📚 Book-Based Learning")
st.markdown("### Upload documents and learn with AI-powered RAG")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "💬 Ask Questions", "🔍 Explore Topics", "📖 Q&A History"])

# Tab 1: Upload Documents
with tab1:
    st.markdown("#### 📤 Upload Learning Materials")
    st.markdown("Upload PDF, TXT, DOCX, or MD files to build your knowledge base")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Upload documents to add to the knowledge base"
        )
    
    with col2:
        category = st.selectbox(
            "Category",
            options=["algebra", "calculus", "geometry", "statistics", "general", "other"],
            help="Categorize your document for better organization"
        )
    
    if st.button("🚀 Upload and Process", type="primary"):
        if uploaded_files:
            upload_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save file temporarily
                temp_path = os.path.join(settings.UPLOAD_DIR, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to vector store
                result = st.session_state.vector_store.add_documents(temp_path, category)
                upload_results.append(result)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            st.markdown("---")
            st.markdown("### ✅ Upload Results")
            
            total_chunks = sum(r["chunks_added"] for r in upload_results)
            successful = sum(1 for r in upload_results if r["success"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Processed", len(upload_results))
            with col2:
                st.metric("Successful", successful)
            with col3:
                st.metric("Total Chunks", total_chunks)
            
            for result in upload_results:
                if result["success"]:
                    st.success(f"✅ {result['file_name']}: {result['chunks_added']} chunks added")
                else:
                    st.error(f"❌ {result.get('file_name', 'Unknown')}: {result['message']}")
        else:
            st.warning("⚠️ Please select files to upload")
    
    # Document Management
    st.markdown("---")
    st.markdown("#### 🗂️ Manage Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if stats["document_names"]:
            doc_to_delete = st.selectbox(
                "Select document to delete",
                options=stats["document_names"]
            )
        else:
            st.info("No documents in the knowledge base yet")
            doc_to_delete = None
    
    with col2:
        st.markdown("")
        st.markdown("")
        if doc_to_delete and st.button("🗑️ Delete Document", type="secondary"):
            if st.session_state.vector_store.delete_document(doc_to_delete):
                st.success(f"✅ Deleted {doc_to_delete}")
                st.rerun()
            else:
                st.error("❌ Failed to delete document")
    
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if st.session_state.vector_store.clear_collection():
            st.success("✅ All documents cleared")
            st.rerun()

# Tab 2: Ask Questions
with tab2:
    st.markdown("#### 💬 Ask Questions About Your Documents")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "Your Question",
            height=100,
            placeholder="Ask anything about your uploaded documents..."
        )
    
    with col2:
        st.markdown("**Filter by Category:**")
        use_category_filter = st.checkbox("Enable filter", key="q_filter")
        if use_category_filter:
            question_category = st.selectbox(
                "Category",
                options=["algebra", "calculus", "geometry", "statistics", "general", "other"],
                label_visibility="collapsed",
                key="q_category"
            )
        else:
            question_category = None
        
        n_context = st.slider(
            "Context chunks",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant chunks to use"
        )
    
    if st.button("🔍 Get Answer", type="primary", key="ask_question"):
        if question:
            if stats["total_chunks"] == 0:
                st.warning("⚠️ Please upload documents first!")
            else:
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.rag_agent.answer_question(
                        question=question,
                        category=question_category,
                        n_context=n_context
                    )
                    
                    # Add to history
                    st.session_state.rag_history.append(result)
                    
                    if result["success"]:
                        st.markdown("---")
                        st.markdown("### 🎯 Answer")
                        
                        # Display answer with LaTeX rendering
                        st.markdown(result["answer"])
                        
                        st.markdown("---")
                        
                        # Sources and metadata
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**📚 Sources Used:**")
                            for source in result["sources"]:
                                st.markdown(f"- {source}")
                        
                        with col2:
                            st.metric("Context Chunks", result["context_used"])
                            st.markdown(f"**Model:** {result['model_used']}")
                    else:
                        st.error(f"❌ {result['answer']}")
        else:
            st.warning("⚠️ Please enter a question")

# Tab 3: Explore Topics
with tab3:
    st.markdown("#### 🔍 Explore Topics in Your Documents")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📖 Explain Concept")
        concept = st.text_input("Enter concept", key="concept_input")
        if st.button("Explain", key="explain_btn"):
            if concept:
                with st.spinner("Generating explanation..."):
                    result = st.session_state.rag_agent.explain_concept(concept)
                    st.session_state.rag_history.append(result)
                    
                    if result["success"]:
                        st.markdown(result["answer"])
                        st.markdown(f"**Sources:** {', '.join(result['sources'])}")
    
    with col2:
        st.markdown("##### 💡 Find Examples")
        topic = st.text_input("Enter topic", key="topic_input")
        if st.button("Find Examples", key="examples_btn"):
            if topic:
                with st.spinner("Finding examples..."):
                    result = st.session_state.rag_agent.find_examples(topic)
                    st.session_state.rag_history.append(result)
                    
                    if result["success"]:
                        st.markdown(result["answer"])
                        st.markdown(f"**Sources:** {', '.join(result['sources'])}")
    
    with col3:
        st.markdown("##### 📝 Summarize Topic")
        summary_topic = st.text_input("Enter topic", key="summary_input")
        if st.button("Summarize", key="summary_btn"):
            if summary_topic:
                with st.spinner("Creating summary..."):
                    result = st.session_state.rag_agent.summarize_topic(summary_topic)
                    st.session_state.rag_history.append(result)
                    
                    if result["success"]:
                        st.markdown(result["answer"])
                        st.markdown(f"**Sources:** {', '.join(result['sources'])}")

# Tab 4: Q&A History
with tab4:
    st.markdown("### 📖 Q&A History")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_history = st.text_input("🔍 Search history", placeholder="Enter keyword...")
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.rag_history = []
            st.success("History cleared!")
            st.rerun()
    
    if st.session_state.rag_history:
        # Filter history
        if search_history:
            filtered_history = [
                h for h in st.session_state.rag_history
                if search_history.lower() in h.get("question", "").lower() or
                   search_history.lower() in h.get("answer", "").lower()
            ]
        else:
            filtered_history = st.session_state.rag_history[::-1]  # Reverse to show latest first
        
        if not filtered_history:
            st.info("No matching history found")
        else:
            for idx, entry in enumerate(filtered_history):
                if entry.get("success"):
                    with st.expander(f"Q{len(filtered_history)-idx}: {entry.get('question', 'Unknown')[:100]}..."):
                        st.markdown(f"**Question:** {entry.get('question', 'N/A')}")
                        st.markdown("**Answer:**")
                        st.markdown(entry.get('answer', 'N/A'))
                        
                        if entry.get('sources'):
                            st.markdown(f"**Sources:** {', '.join(entry['sources'])}")
                        
                        st.markdown(f"**Context Used:** {entry.get('context_used', 0)} chunks")
                        st.markdown(f"**Model:** {entry.get('model_used', 'N/A')}")
    else:
        st.info("No Q&A history yet. Start asking questions!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Book-Based Learning System | RAG-Powered Knowledge Base</p>
    </div>
    """,
    unsafe_allow_html=True
)
