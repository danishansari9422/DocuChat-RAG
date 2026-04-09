"""
DocuChat-RAG Streamlit Application

A Streamlit app for chatting with PDF documents using RAG.
"""

import streamlit as st
import os
import tempfile
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from rag.loader import PDFLoader
from rag.chunking import TextChunker
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import DocumentRetriever
from rag.qa_chain import QAChain
from utils.session import SessionManager


def initialize_session():
    """Initialize Streamlit session state."""
    session_manager = SessionManager()
    session_manager.initialize_session()
    return session_manager


def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """
    Process uploaded PDF file and create vector store.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dictionary with processing results
    """
    if uploaded_file is None:
        return {'success': False, 'error': 'No file uploaded'}
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Initialize components
        loader = PDFLoader()
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore(persist_directory="./storage")
        
        # Load PDF
        with st.spinner("Loading PDF..."):
            documents = loader.load_pdf(tmp_file_path)
            doc_info = loader.get_document_info(documents)
        
        # Chunk documents
        with st.spinner("Splitting text into chunks..."):
            chunks = chunker.split_documents(documents)
            chunking_stats = chunker.get_chunking_stats(chunks)
        
        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            texts = [chunk.page_content for chunk in chunks]
            embeddings = embedding_manager.get_embeddings(texts)
        
        # Add to vector store
        with st.spinner("Creating vector database..."):
            vector_store.clear_collection()  # Clear previous data
            vector_store.add_documents(chunks, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            'success': True,
            'doc_info': doc_info,
            'chunking_stats': chunking_stats,
            'vector_store_stats': vector_store.get_collection_stats(),
            'model_info': embedding_manager.get_model_info()
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        # Provide more specific error information
        error_str = str(e)
        if "FileNotFoundError" in error_str:
            user_error = "PDF file not found. Please check if the file was uploaded correctly."
        elif "empty" in error_str.lower():
            user_error = "The uploaded PDF file appears to be empty."
        elif "large" in error_str.lower():
            user_error = "The PDF file is too large. Please use a file smaller than 50MB."
        elif "encrypted" in error_str.lower() or "password" in error_str.lower():
            user_error = "The PDF file is password-protected. Please use an unprotected PDF."
        elif "corrupted" in error_str.lower() or "damaged" in error_str.lower():
            user_error = "The PDF file appears to be corrupted or damaged."
        elif "No content extracted" in error_str:
            user_error = "No text could be extracted from the PDF. It might be an image-only PDF."
        elif "ModuleNotFoundError" in error_str:
            user_error = "Missing required Python modules. Please run: pip install -r requirements.txt"
        else:
            user_error = f"Document processing failed: {error_str}"
        
        return {'success': False, 'error': user_error}


def initialize_rag_components(model_name: str = "gemini-2.5-flash"):
    """Initialize RAG components for Q&A."""
    try:
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore(persist_directory="./storage")
        retriever = DocumentRetriever(vector_store, embedding_manager)
        qa_chain = QAChain(retriever, model_name)
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Failed to initialize RAG components: {str(e)}")
        return None


def display_chat_history():
    """Display chat history in the main chat area."""
    session_manager = SessionManager()
    chat_history = session_manager.get_chat_history()
    
    for message in chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
                
                # Display citations if available
                if message['citations']:
                    st.write("**Sources:**")
                    for citation in message['citations']:
                        st.write(f"- Page {citation['page']}: {citation['snippet']}")


def display_sidebar_chat_history():
    """Display chat history in the sidebar."""
    session_manager = SessionManager()
    chat_history = session_manager.get_chat_history()
    
    if not chat_history:
        st.write("No chat history yet")
        return
    
    st.write("**Chat History:**")
    for i, message in enumerate(chat_history[-10:], 1):  # Show last 10 messages
        role_icon = "U" if message['role'] == 'user' else "A"
        content_preview = message['content'][:50] + "..." if len(message['content']) > 50 else message['content']
        st.write(f"{i}. {role_icon}: {content_preview}")


def main():
    """Main application function."""
    st.set_page_config(
        page_title="DocuChat-RAG",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    session_manager = initialize_session()
    
    # App title and description
    st.title("DocuChat-RAG")
    st.write("Upload a PDF document and chat with it using AI-powered RAG (Retrieval-Augmented Generation)")
    
    # Sidebar
    with st.sidebar:
        st.header("Document & Settings")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF file to chat with"
        )
        
        # Model selection
        st.divider()
        st.subheader("AI Model Settings")
        
        # Check if API key is set
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("⚠️ GEMINI_API_KEY not found in .env file")
            st.info("Please add your Gemini API key to the .env file")
        else:
            st.success("✅ Gemini API key configured")
        
        # Model selection
        model_options = [
            "gemini-2.5-flash"
        ]
        
        selected_model = st.selectbox(
            "Select Gemini Model:",
            options=model_options,
            index=0,
            help="Gemini 2.5 Flash model for answer generation"
        )
        
        # Store selected model in session state
        st.session_state.selected_model = selected_model
        
        # Process file button
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    result = process_uploaded_file(uploaded_file)
                
                if result['success']:
                    session_manager.set_current_file({
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'info': result['doc_info']
                    })
                    session_manager.set_vector_store_ready(True)
                    session_manager.set_processing_complete(True)
                    
                    st.success("Document processed successfully!")
                    
                    # Display processing info
                    with st.expander("Processing Details"):
                        st.write("**Document Info:**")
                        st.json(result['doc_info'])
                        
                        st.write("**Chunking Stats:**")
                        st.json(result['chunking_stats'])
                        
                        st.write("**Vector Store Stats:**")
                        st.json(result['vector_store_stats'])
                        
                        st.write("**Model Info:**")
                        st.json(result['model_info'])
                else:
                    st.error(f"Error processing document: {result['error']}")
                    session_manager.set_processing_complete(False)
        
        # Display current file info
        current_file = session_manager.get_current_file()
        if current_file:
            st.write("**Current Document:**")
            st.write(f"Name: {current_file['name']}")
            st.write(f"Size: {current_file['size']:,} bytes")
            
            if 'info' in current_file:
                st.write(f"Pages: {current_file['info'].get('total_pages', 'Unknown')}")
        
        # Clear buttons
        st.divider()
        if st.button("Clear Chat History"):
            session_manager.clear_chat_history()
            st.rerun()
        
        if st.button("Reset Everything"):
            session_manager.reset_session()
            # Clear vector store
            try:
                vector_store = VectorStore(persist_directory="./storage")
                vector_store.clear_collection()
            except:
                pass
            st.rerun()
        
        # Evaluation section
        st.divider()
        st.subheader("Evaluation")
        
        if session_manager.is_vector_store_ready():
            if st.button("Run Evaluation", type="primary"):
                # Initialize evaluator
                model_name = st.session_state.get('selected_model', 'gemini-2.5-flash')
                qa_chain = initialize_rag_components(model_name)
                
                if qa_chain:
                    from eval.evaluator import RAGEvaluator
                    evaluator = RAGEvaluator(qa_chain)
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total, question):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {current + 1}/{total}: {question[:50]}...")
                    
                    # Run evaluation
                    with st.spinner("Running evaluation..."):
                        results = evaluator.run_evaluation(progress_callback)
                        # Store results in session state
                        st.session_state.evaluation_results = results
                        st.session_state.evaluator = evaluator
                        st.session_state.evaluation_complete = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("Evaluation complete!")
                    st.success(f"Evaluation completed! {len(results)} questions processed.")
                    
                    # Show initial metrics
                    metrics = evaluator.calculate_accuracy()
                    st.info(f"Initial Results: {metrics['correct_rate']} accuracy")
                else:
                    st.error("Failed to initialize QA chain for evaluation")
        else:
            st.info("Please upload and process a document before running evaluation")
        
        # Chat history in sidebar
        st.divider()
        display_sidebar_chat_history()
    
    # Main chat interface
    if session_manager.is_vector_store_ready():
        # Display chat history
        display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document"):
            # Add user message to history
            session_manager.add_message("user", prompt)
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Generate response
            with st.spinner("Thinking..."):
                # Get selected model from session state or use default
                model_name = st.session_state.get('selected_model', 'gemini-2.5-flash')
                qa_chain = initialize_rag_components(model_name)
                if qa_chain:
                    chat_history = session_manager.get_chat_history()
                    result = qa_chain.answer_question(prompt, chat_history)
                    
                    # Add assistant response to history
                    session_manager.add_message(
                        "assistant", 
                        result['answer'], 
                        result['citations']
                    )
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(result['answer'])
                        
                        # Display citations
                        if result['citations']:
                            st.write("**Sources:**")
                            for citation in result['citations']:
                                st.write(f"- Page {citation['page']}: {citation['snippet']}")
                        
                        # Display context usage info
                        if not result['context_used']:
                            st.info("No relevant information found in the document.")
                else:
                    st.error("Failed to initialize Q&A system. Please try reprocessing the document.")
            
            # Rerun to update the display
            st.rerun()
    
    # Display evaluation results if available
    if st.session_state.get('evaluation_complete', False):
        st.divider()
        st.header("Evaluation Results")
        
        evaluator = st.session_state.get('evaluator')
        results = st.session_state.get('evaluation_results', [])
        
        if evaluator and results:
            # Display metrics
            metrics = evaluator.calculate_accuracy()
            summary_stats = evaluator.get_summary_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
            with col2:
                st.metric("Correct", f"{metrics['total_correct']}/{metrics['total_questions']}")
            with col3:
                st.metric("Context Usage", summary_stats['context_usage_rate'])
            with col4:
                st.metric("Avg Answer Length", summary_stats['avg_answer_length'])
            
            # Display results table
            st.subheader("Detailed Results")
            df = evaluator.get_results_dataframe()
            
            if not df.empty:
                # Side-by-side comparison interface
                st.write("Compare RAG LLM answers with Ground Truth:")
                
                for i, result in enumerate(results):
                    with st.container():
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write(f"**Q{i+1}:** {result['question'][:100]}...")
                            st.write("---")
                            
                            # Ground Truth (Expected Answer)
                            st.write("**📋 Ground Truth:**")
                            st.info(result['expected_answer'])
                            st.write("")
                            
                            # RAG LLM Answer
                            st.write("**🤖 RAG LLM Answer:**")
                            st.warning(result['model_answer'][:500] + "..." if len(result['model_answer']) > 500 else result['model_answer'])
                            st.write("")
                            
                            # Marking buttons
                            col_correct, col_wrong = st.columns(2)
                            with col_correct:
                                if st.button(f"✅ Correct", key=f"correct_{i}", help="Mark RAG answer as correct"):
                                    evaluator.update_result(i, True, "Matches ground truth")
                                    st.rerun()
                            with col_wrong:
                                if st.button(f"❌ Wrong", key=f"wrong_{i}", help="Mark RAG answer as wrong"):
                                    evaluator.update_result(i, False, "Does not match ground truth")
                                    st.rerun()
                        
                        # Show current status
                        if result['correct'] is True:
                            st.success("✅ Marked as Correct")
                        elif result['correct'] is False:
                            st.error("❌ Marked as Wrong")
                        else:
                            st.info("⏸️ Not marked")
                        
                        st.divider()
                
                # Calculate and display accuracy
                metrics = evaluator.calculate_accuracy()
                st.divider()
                st.subheader("Accuracy Metrics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
                with col2:
                    st.metric("Correct", f"{metrics['total_correct']}/{metrics['total_questions']}")
                with col3:
                    st.metric("Wrong", f"{metrics['total_incorrect']}/{metrics['total_questions']}")
                
                # Export functionality
                st.divider()
                if st.button("Export Results to CSV"):
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"evaluation_results_{timestamp}.csv"
                    if evaluator.export_results(filename):
                        st.success(f"Results exported to {filename}")
                    else:
                        st.error("Failed to export results")
            
            # Clear evaluation button
            if st.button("Clear Evaluation Results"):
                st.session_state.evaluation_results = []
                st.session_state.evaluator = None
                st.session_state.evaluation_complete = False
                st.rerun()
    
    else:
        # Initial state - no document uploaded
        st.info("Please upload a PDF document to get started. The document will be processed and you'll be able to ask questions about it.")
        
        # Display instructions
        with st.expander("How to use DocuChat-RAG"):
            st.write("""
            1. **Upload a PDF**: Use the file uploader in the sidebar to upload a PDF document
            2. **Process the document**: Click "Process Document" to analyze and index the PDF
            3. **Ask questions**: Once processed, you can ask questions about the document content
            4. **Get answers with citations**: The AI will provide answers with page number references
            
            **Features:**
            - RAG (Retrieval-Augmented Generation) for accurate answers
            - Citations with page numbers and text snippets
            - Persistent chat history
            - Local processing (no external APIs required)
            """)
        
        # Display processing status
        if session_manager.is_processing_complete() is False and uploaded_file is not None:
            st.info("Ready to upload a PDF document. Use the file uploader in the sidebar to get started.")


if __name__ == "__main__":
    main()
