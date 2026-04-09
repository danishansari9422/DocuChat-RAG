"""
Q&A Chain Module

Handles question answering with RAG pipeline.
"""

from typing import List, Dict, Any
from .retriever import DocumentRetriever
from .llm import GeminiLLM


class QAChain:
    """Handles question answering using retrieved documents."""
    
    def __init__(self, retriever: DocumentRetriever, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Q&A chain.
        
        Args:
            retriever: DocumentRetriever instance for document retrieval
            model_name: Gemini model to use
        """
        self.retriever = retriever
        self.llm = GeminiLLM(model_name)
        self.system_prompt = """You are a helpful assistant. Answer ONLY from the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer with citations like:
(Source: Page X)"""
    
    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format chat history for inclusion in prompt.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return ""
        
        history_parts = []
        for message in chat_history[-5:]:  # Include last 5 messages for context
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'user':
                history_parts.append(f"User: {content}")
            else:
                history_parts.append(f"Assistant: {content}")
        
        return "\n".join(history_parts)
    
    def answer_question(
        self, 
        question: str, 
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User question
            chat_history: Previous conversation history
            
        Returns:
            Dictionary with answer and citations
        """
        if not question.strip():
            return {
                'answer': 'Please ask a question.',
                'citations': [],
                'context_used': False
            }
        
        chat_history = chat_history or []
        
        try:
            # Retrieve relevant documents
            retrieval_result = self.retriever.search_with_citations(question, top_k=4)
            context = retrieval_result['context']
            citations = retrieval_result['citations']
            
            # Format the prompt
            formatted_prompt = self.system_prompt.format(
                context=context,
                question=question
            )
            
            # Add chat history if available
            if chat_history:
                history_text = self.format_chat_history(chat_history)
                if history_text:
                    formatted_prompt = f"""Chat History:
{history_text}

{formatted_prompt}"""
            
            # Use Gemini LLM for answer generation
            llm_result = self.llm.generate(context, question, chat_history)
            answer = llm_result['answer']
            
            # Use Gemini's sources if available, otherwise fall back to our citations
            sources = llm_result.get('sources', citations)
            
            return {
                'answer': answer,
                'citations': sources,
                'context_used': len(citations) > 0,
                'retrieved_count': retrieval_result['retrieved_count']
            }
            
        except Exception as e:
            return {
                'answer': f'Sorry, I encountered an error: {str(e)}',
                'citations': [],
                'context_used': False,
                'error': str(e)
            }
    
    def _generate_simple_answer(
        self, 
        question: str, 
        context: str, 
        citations: List[Dict[str, str]]
    ) -> str:
        """
        Generate a simple answer based on the context.
        
        Args:
            question: User question
            context: Retrieved context
            citations: Citation information
            
        Returns:
            Generated answer
        """
        # This is a simplified answer generation
        # In a real implementation, you would use an LLM
        
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Check if context contains relevant information
        if not context or context == "No relevant documents found.":
            return "I don't have enough information to answer this question from the provided documents."
        
        # Simple keyword-based response (placeholder for real LLM)
        if any(word in context_lower for word in ['what', 'who', 'when', 'where', 'why', 'how']):
            answer = f"Based on the provided documents, I can see information related to your question. "
            
            # Add citations
            if citations:
                pages = [str(c['page']) for c in citations[:2]]  # Use first 2 citations
                answer += f"(Source: Page {', '.join(pages)})"
            
            return answer
        
        # Default response
        answer = "I found relevant information in the documents. "
        if citations:
            pages = [str(c['page']) for c in citations[:2]]
            answer += f"(Source: Page {', '.join(pages)})"
        
        return answer
    
    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get information about the Q&A chain.
        
        Returns:
            Dictionary with chain information
        """
        return {
            'retriever_type': type(self.retriever).__name__,
            'system_prompt_length': len(self.system_prompt),
            'supports_chat_history': True
        }
