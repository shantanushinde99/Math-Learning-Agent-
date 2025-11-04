"""
Vector Store Manager for RAG Implementation
Handles document embedding, storage, and retrieval using ChromaDB
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from config.settings import settings
from src.utils.logger import app_logger


class VectorStoreManager:
    """Manages vector database operations for RAG"""
    
    def __init__(self):
        """Initialize the vector store with ChromaDB"""
        self.db_path = settings.VECTOR_DB_PATH
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embeddings model (free, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Get or create collections
        self.math_collection = self._get_or_create_collection("math_documents")
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        app_logger.info("✅ Vector Store Manager initialized successfully")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            collection = self.client.get_collection(name)
            app_logger.info(f"📂 Loaded existing collection: {name}")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            app_logger.info(f"📂 Created new collection: {name}")
        
        return collection
    
    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document based on its file type
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks with metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # Choose appropriate loader based on file type
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load and split documents
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            app_logger.info(f"📄 Loaded {len(chunks)} chunks from {os.path.basename(file_path)}")
            
            # Convert to our format
            processed_chunks = []
            for idx, chunk in enumerate(chunks):
                processed_chunks.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            return processed_chunks
            
        except Exception as e:
            app_logger.error(f"❌ Error loading document {file_path}: {str(e)}")
            raise
    
    def add_documents(self, file_path: str, category: str = "general") -> Dict[str, Any]:
        """
        Add documents to the vector store
        
        Args:
            file_path: Path to the document
            category: Category/tag for the document
            
        Returns:
            Dictionary with upload statistics
        """
        try:
            # Load and process document
            chunks = self.load_document(file_path)
            
            if not chunks:
                return {
                    "success": False,
                    "message": "No content extracted from document",
                    "chunks_added": 0
                }
            
            # Prepare data for ChromaDB
            texts = [chunk["text"] for chunk in chunks]
            metadatas = []
            ids = []
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare metadata and IDs
            base_id = f"{category}_{os.path.basename(file_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            for idx, chunk in enumerate(chunks):
                chunk_metadata = chunk["metadata"]
                chunk_metadata["category"] = category
                metadatas.append(chunk_metadata)
                ids.append(f"{base_id}_chunk_{idx}")
            
            # Add to collection
            self.math_collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            result = {
                "success": True,
                "message": f"Successfully added {len(chunks)} chunks from document",
                "chunks_added": len(chunks),
                "file_name": os.path.basename(file_path),
                "category": category
            }
            
            app_logger.info(f"✅ Added document: {result['file_name']} ({result['chunks_added']} chunks)")
            
            return result
            
        except Exception as e:
            app_logger.error(f"❌ Error adding document: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "chunks_added": 0
            }
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            category: Optional category filter
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Prepare where clause for filtering
            where_clause = None
            if category:
                where_clause = {"category": category}
            
            # Search in collection
            results = self.math_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for idx in range(len(results['documents'][0])):
                    formatted_results.append({
                        "text": results['documents'][0][idx],
                        "metadata": results['metadatas'][0][idx],
                        "distance": results['distances'][0][idx] if 'distances' in results else None
                    })
            
            app_logger.info(f"🔍 Search returned {len(formatted_results)} results for query: {query[:50]}...")
            
            return formatted_results
            
        except Exception as e:
            app_logger.error(f"❌ Error searching: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            count = self.math_collection.count()
            
            # Get sample of documents to analyze categories
            if count > 0:
                sample = self.math_collection.get(limit=min(1000, count))
                categories = set()
                sources = set()
                
                for metadata in sample['metadatas']:
                    if 'category' in metadata:
                        categories.add(metadata['category'])
                    if 'source' in metadata:
                        sources.add(metadata['source'])
                
                return {
                    "total_chunks": count,
                    "categories": list(categories),
                    "unique_documents": len(sources),
                    "document_names": list(sources)
                }
            else:
                return {
                    "total_chunks": 0,
                    "categories": [],
                    "unique_documents": 0,
                    "document_names": []
                }
                
        except Exception as e:
            app_logger.error(f"❌ Error getting stats: {str(e)}")
            return {
                "total_chunks": 0,
                "categories": [],
                "unique_documents": 0,
                "document_names": []
            }
    
    def delete_document(self, document_name: str) -> bool:
        """Delete all chunks from a specific document"""
        try:
            # Get all IDs for this document
            all_data = self.math_collection.get()
            ids_to_delete = []
            
            for idx, metadata in enumerate(all_data['metadatas']):
                if metadata.get('source') == document_name:
                    ids_to_delete.append(all_data['ids'][idx])
            
            if ids_to_delete:
                self.math_collection.delete(ids=ids_to_delete)
                app_logger.info(f"🗑️ Deleted {len(ids_to_delete)} chunks from {document_name}")
                return True
            else:
                app_logger.warning(f"⚠️ No chunks found for document: {document_name}")
                return False
                
        except Exception as e:
            app_logger.error(f"❌ Error deleting document: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection("math_documents")
            self.math_collection = self._get_or_create_collection("math_documents")
            app_logger.info("🗑️ Cleared entire collection")
            return True
        except Exception as e:
            app_logger.error(f"❌ Error clearing collection: {str(e)}")
            return False
