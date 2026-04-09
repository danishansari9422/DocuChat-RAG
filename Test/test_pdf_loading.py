"""
Test script to check PDF loading functionality
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag.loader import PDFLoader
    from rag.chunking import TextChunker
    from rag.embeddings import EmbeddingManager
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install missing dependencies: pip install -r requirements.txt")
    sys.exit(1)

def test_pdf_loading():
    """Test PDF loading with the sample file"""
    pdf_path = "../SampleData/Rethinking_Retrieval_From_Traditional_Retrieval_Au.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return False
    
    print(f"Testing PDF: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    
    try:
        # Initialize loader
        loader = PDFLoader()
        
        # Load PDF
        print("Loading PDF...")
        documents = loader.load_pdf(pdf_path)
        
        print(f"PDF loaded successfully!")
        print(f"Number of pages/documents: {len(documents)}")
        
        if documents:
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"Total characters extracted: {total_chars:,}")
            print(f"Average characters per page: {total_chars // len(documents):,}")
            
            # Show first few characters of first page
            if documents[0].page_content:
                preview = documents[0].page_content[:200] + "..." if len(documents[0].page_content) > 200 else documents[0].page_content
                print(f"First page preview: {preview}")
            
            # Test chunking
            print("\nTesting chunking...")
            chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
            chunks = chunker.split_documents(documents)
            print(f"Chunking successful!")
            print(f"Number of chunks: {len(chunks)}")
            
            # Test embeddings (just a small sample)
            print("\nTesting embeddings (first 3 chunks)...")
            embedding_manager = EmbeddingManager()
            sample_texts = [chunk.page_content for chunk in chunks[:3]]
            embeddings = embedding_manager.get_embeddings(sample_texts)
            print(f"Embeddings successful!")
            print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
            
            return True
        else:
            print("No documents extracted from PDF")
            return False
            
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        return False

if __name__ == "__main__":
    print("Testing DocuChat-RAG PDF Loading")
    print("=" * 50)
    
    success = test_pdf_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed! Your PDF should work with the RAG system.")
    else:
        print("Tests failed. Check the error messages above.")
        print("Try:")
        print("   1. Installing missing dependencies: pip install -r requirements.txt")
        print("   2. Using a different PDF file")
        print("   3. Checking if the PDF is password-protected or corrupted")
