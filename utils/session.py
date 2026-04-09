"""
Session Management Module

Handles chat history and session state management.
"""

from typing import List, Dict, Any
import streamlit as st


class SessionManager:
    """Manages Streamlit session state and chat history."""
    
    def __init__(self):
        """Initialize session manager."""
        self.session_state = st.session_state
    
    def initialize_session(self):
        """Initialize session state variables."""
        if 'chat_history' not in self.session_state:
            self.session_state.chat_history = []
        
        if 'current_file' not in self.session_state:
            self.session_state.current_file = None
        
        if 'vector_store_ready' not in self.session_state:
            self.session_state.vector_store_ready = False
        
        if 'processing_complete' not in self.session_state:
            self.session_state.processing_complete = False
    
    def add_message(self, role: str, content: str, citations: List[Dict[str, str]] = None):
        """
        Add a message to chat history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            citations: List of citations for assistant messages
        """
        message = {
            'role': role,
            'content': content,
            'citations': citations or []
        }
        
        self.session_state.chat_history.append(message)
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the current chat history.
        
        Returns:
            List of chat messages
        """
        return self.session_state.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self.session_state.chat_history = []
    
    def set_current_file(self, file_info: Dict[str, Any]):
        """
        Set the current uploaded file information.
        
        Args:
            file_info: Dictionary with file information
        """
        self.session_state.current_file = file_info
    
    def get_current_file(self) -> Dict[str, Any]:
        """
        Get the current file information.
        
        Returns:
            Current file information dictionary
        """
        return self.session_state.current_file
    
    def set_vector_store_ready(self, ready: bool):
        """
        Set vector store readiness status.
        
        Args:
            ready: Whether vector store is ready
        """
        self.session_state.vector_store_ready = ready
    
    def is_vector_store_ready(self) -> bool:
        """
        Check if vector store is ready.
        
        Returns:
            True if vector store is ready
        """
        return self.session_state.vector_store_ready
    
    def set_processing_complete(self, complete: bool):
        """
        Set processing completion status.
        
        Args:
            complete: Whether processing is complete
        """
        self.session_state.processing_complete = complete
    
    def is_processing_complete(self) -> bool:
        """
        Check if processing is complete.
        
        Returns:
            True if processing is complete
        """
        return self.session_state.processing_complete
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        chat_history = self.get_chat_history()
        user_messages = [msg for msg in chat_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in chat_history if msg['role'] == 'assistant']
        
        return {
            'total_messages': len(chat_history),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'current_file': self.get_current_file(),
            'vector_store_ready': self.is_vector_store_ready(),
            'processing_complete': self.is_processing_complete()
        }
    
    def reset_session(self):
        """Reset the entire session state."""
        self.clear_chat_history()
        self.set_current_file(None)
        self.set_vector_store_ready(False)
        self.set_processing_complete(False)
