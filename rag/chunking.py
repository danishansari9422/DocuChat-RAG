"""
Text Chunking Module

Handles splitting documents into chunks for better retrieval.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    """Handles text splitting with metadata preservation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use RecursiveCharacterTextSplitter for intelligent splitting
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        # Split documents
        chunks = self.splitter.split_documents(documents)
        
        # Ensure each chunk has proper metadata
        for i, chunk in enumerate(chunks):
            # Add chunk index to metadata
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            # Preserve original metadata
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = 'Unknown'
        
        return chunks
    
    def get_chunking_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunking process.
        
        Args:
            chunks: List of chunked Document objects
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / total_chunks
        min_chunk_size = min(len(chunk.page_content) for chunk in chunks)
        max_chunk_size = max(len(chunk.page_content) for chunk in chunks)
        
        return {
            'total_chunks': total_chunks,
            'average_chunk_size': round(avg_chunk_size, 2),
            'min_chunk_size': min_chunk_size,
            'max_chunk_size': max_chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
