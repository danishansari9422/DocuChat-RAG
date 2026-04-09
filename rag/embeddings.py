"""
Embeddings Module

Handles text embeddings using sentence-transformers.
"""

import os
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib


class EmbeddingManager:
    """Manages text embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
    def load_model(self):
        """Load the sentence-transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        self.load_model()
        
        try:
            # Get embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert to list format
            return embeddings.tolist()
            
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Single embedding vector
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def get_text_hash(self, text: str) -> str:
        """
        Generate a hash for text to identify duplicates.
        
        Args:
            text: Text to hash
            
        Returns:
            SHA256 hash of the text
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'model_loaded': self.model is not None
        }
