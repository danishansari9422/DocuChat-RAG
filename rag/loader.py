"""
PDF Loader Module

Handles loading PDF documents and extracting text with metadata.
"""

import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """Handles PDF loading and text extraction with metadata."""
    
    def __init__(self):
        self.loader = None
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF and extract text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with text and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF loading fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"PDF file is empty: {file_path}")
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError(f"PDF file too large ({file_size/1024/1024:.1f}MB). Maximum size is 50MB.")
        
        try:
            # Use PyPDFLoader to load PDF with metadata
            self.loader = PyPDFLoader(file_path)
            documents = self.loader.load()
            
            if not documents:
                raise ValueError(f"No content extracted from PDF: {file_path}")
            
            # Add additional metadata
            for doc in documents:
                doc.metadata['source'] = os.path.basename(file_path)
                doc.metadata['file_path'] = file_path
                doc.metadata['file_size'] = file_size
                
            return documents
            
        except Exception as e:
            # Provide more specific error information
            error_msg = f"Failed to load PDF '{os.path.basename(file_path)}': {str(e)}"
            if "encrypted" in str(e).lower():
                error_msg += " (PDF may be password-protected)"
            elif "corrupted" in str(e).lower() or "damaged" in str(e).lower():
                error_msg += " (PDF file may be corrupted)"
            elif "permission" in str(e).lower():
                error_msg += " (Insufficient permissions to read PDF)"
            
            raise Exception(error_msg)
    
    def get_document_info(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get information about loaded documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {}
        
        total_pages = len(documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        return {
            'total_pages': total_pages,
            'total_characters': total_chars,
            'source': documents[0].metadata.get('source', 'Unknown'),
            'file_path': documents[0].metadata.get('file_path', 'Unknown')
        }
