"""
Gemini LLM Module

Handles integration with Google Gemini for answer generation.
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional


class GeminiLLM:
    """Handles Gemini LLM for RAG answer generation."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini LLM.
        
        Args:
            model_name: Gemini model to use (gemini-1.5-flash or gemini-2.0-flash)
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def generate(
        self, 
        context: str, 
        question: str, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using Gemini with context and question.
        
        Args:
            context: Retrieved context from vector database
            question: User's question
            chat_history: Previous chat messages for context
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Format chat history if provided
            history_text = ""
            if chat_history:
                history_text = "\n\nChat History:\n"
                for msg in chat_history[-5:]:  # Keep last 5 messages
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    history_text += f"{role.title()}: {content}\n"
            
            # Create prompt
            prompt = self._create_prompt(context, question, history_text)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Extract answer and sources
            answer = response.text
            
            return {
                'answer': answer,
                'sources': self._extract_sources(context),
                'model': self.model_name,
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'model': self.model_name,
                'success': False
            }
    
    def _create_prompt(self, context: str, question: str, history_text: str = "") -> str:
        """
        Create a comprehensive prompt for Gemini.
        
        Args:
            context: Retrieved document context
            question: User's question
            history_text: Formatted chat history
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

STRICT RULES:
- Answer ONLY using the provided context below
- Do NOT hallucinate or make up information
- If the answer is not in the context, say "I don't know based on the provided documents"
- Provide specific, detailed answers when possible
- Include citations in your answers using (Source: Page X) format

{history_text}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        return prompt
    
    def _extract_sources(self, context: str) -> List[str]:
        """
        Extract source information from context.
        
        Args:
            context: Formatted context string
            
        Returns:
            List of source references
        """
        sources = []
        lines = context.split('\n')
        
        for line in lines:
            if line.startswith('Page '):
                # Extract page number and snippet
                page_num = line.split(':')[0].replace('Page ', '')
                snippet = line.split(':', 1)[1].strip() if ':' in line else ""
                if snippet:
                    sources.append(f"Page {page_num}: {snippet[:100]}...")
        
        return sources
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'provider': 'Google Generative AI',
            'api_key_set': bool(os.getenv("GEMINI_API_KEY"))
        }
