"""
Test script to verify all imports work correctly.
"""

def test_imports():
    """Test importing all required modules."""
    try:
        print("Testing imports...")
        
        # Test core imports
        import streamlit as st
        print("✓ streamlit")
        
        # Test langchain imports
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        print("✓ langchain")
        
        # Test sentence-transformers
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers")
        
        # Test chromadb
        import chromadb
        print("✓ chromadb")
        
        # Test pypdf
        import pypdf
        print("✓ pypdf")
        
        # Test numpy
        import numpy as np
        print("✓ numpy")
        
        # Test our custom modules
        from rag.loader import PDFLoader
        from rag.chunking import TextChunker
        from rag.embeddings import EmbeddingManager
        from rag.vector_store import VectorStore
        from rag.retriever import DocumentRetriever
        from rag.qa_chain import QAChain
        print("✓ rag modules")
        
        from utils.session import SessionManager
        print("✓ utils modules")
        
        print("\nAll imports successful!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
