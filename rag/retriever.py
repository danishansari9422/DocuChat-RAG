
from typing import List, Dict, Any
from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class DocumentRetriever:
    """Handles document retrieval with similarity search."""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance for document storage
            embedding_manager: EmbeddingManager for query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.default_top_k = 4
    
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        if not query.strip():
            return []
        
        top_k = top_k or self.default_top_k
        
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.get_embedding(query)
            
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to retrieve documents: {str(e)}")
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            page = doc.get('page', 1)
            source = doc.get('source', 'Unknown')
            content = doc.get('document', '')
            
            # Format each document with citation info
            context_part = f"[Document {i}] (Page {page}, Source: {source})\n{content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def get_citations(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract citation information from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for doc in retrieved_docs:
            page = doc.get('page', 1)
            source = doc.get('source', 'Unknown')
            content = doc.get('document', '')
            
            # Create a short snippet (first 100 characters)
            snippet = content[:100].strip()
            if len(content) > 100:
                snippet += "..."
            
            citations.append({
                'page': page,
                'source': source,
                'snippet': snippet
            })
        
        return citations
    
    def search_with_citations(
        self, 
        query: str, 
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents and return both context and citations.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with context and citations
        """
        retrieved_docs = self.retrieve_documents(query, top_k)
        
        context = self.format_context(retrieved_docs)
        citations = self.get_citations(retrieved_docs)
        
        return {
            'context': context,
            'citations': citations,
            'retrieved_count': len(retrieved_docs)
        }
