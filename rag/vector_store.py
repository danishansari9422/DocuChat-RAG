"""
Vector Store Module

Handles ChromaDB vector storage and retrieval.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document


class VectorStore:
    """Manages ChromaDB vector storage for document chunks."""
    
    def __init__(self, persist_directory: str = "./storage"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = "docuchat_documents"
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self.client is None:
            try:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=False
                    )
                )
            except Exception as e:
                raise Exception(f"Failed to initialize ChromaDB client: {str(e)}")
        
        return self.client
    
    def _get_collection(self):
        """Get or create collection."""
        if self.collection is None:
            client = self._get_client()
            
            try:
                # Use a simple embedding function (we'll provide our own embeddings)
                self.collection = client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_functions.DefaultEmbeddingFunction(),
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                raise Exception(f"Failed to get/create collection: {str(e)}")
        
        return self.collection
    
    def add_documents(self, chunks: List[Document], embeddings: List[List[float]]):
        """
        Add document chunks with embeddings to the vector store.
        
        Args:
            chunks: List of Document objects
            embeddings: List of embedding vectors
        """
        if not chunks or not embeddings:
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        collection = self._get_collection()
        
        try:
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]
            documents = [chunk.page_content for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = chunk.metadata.copy()
                # Ensure page number is included
                if 'page' not in metadata and 'source' in metadata:
                    metadata['page'] = metadata.get('page', 1)
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
        except Exception as e:
            raise Exception(f"Failed to add documents to vector store: {str(e)}")
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return
            
        Returns:
            List of search results with documents and metadata
        """
        collection = self._get_collection()
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'page': results['metadatas'][0][i].get('page', 1),
                        'source': results['metadatas'][0][i].get('source', 'Unknown')
                    })
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Failed to perform similarity search: {str(e)}")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        collection = self._get_collection()
        
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            self._get_collection()
        except Exception as e:
            raise Exception(f"Failed to clear collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        collection = self._get_collection()
        
        try:
            count = collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            return {'error': str(e)}
