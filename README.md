# DocuChat-RAG

A Streamlit application for chatting with PDF documents using Retrieval-Augmented Generation (RAG) with Google Gemini LLM integration and comprehensive evaluation system.

## Features

- **PDF Upload & Processing**: Upload PDF documents and process them for intelligent querying
- **Google Gemini LLM Integration**: High-quality answer generation using Gemini 2.5 Flash
- **RAG-powered Q&A**: Get accurate answers based on document content
- **Citations**: Answers include page numbers and text snippets for verification
- **Persistent Chat History**: Maintain conversation context throughout the session
- **Comprehensive Evaluation System**: Test RAG performance with 5 predefined questions
- **Side-by-Side Comparison**: Compare RAG answers with ground truth for manual evaluation
- **Real-time Accuracy Metrics**: Track performance and calculate accuracy based on manual marking
- **Export Functionality**: Export evaluation results to CSV
- **Local Processing**: Everything runs locally without requiring external APIs
- **Clean UI**: Modern, intuitive interface similar to ChatGPT

## Project Structure

```
DocuChat-RAG/
|-- app.py                 # Main Streamlit application
|-- rag/                   # RAG pipeline components
|   |-- loader.py          # PDF loading with PyPDFLoader
|   |-- chunking.py        # Text splitting with RecursiveCharacterTextSplitter
|   |-- embeddings.py      # Text embeddings using sentence-transformers
|   |-- vector_store.py    # ChromaDB vector storage
|   |-- retriever.py       # Document retrieval and similarity search
|   |-- qa_chain.py        # Q&A chain with Gemini LLM integration
|   |-- llm.py           # Gemini LLM wrapper for answer generation
|-- eval/                  # Evaluation system
|   |-- evaluator.py       # RAG performance evaluation with 5 test questions
|-- utils/                 # Utility modules
|   |-- session.py         # Session state and chat history management
|-- storage/               # Vector database persistence
|-- .env                   # Environment variables for API keys
|-- .gitignore             # Git ignore file
|-- requirements.txt       # Python dependencies
|-- README.md             # This file
```

## Setup Instructions

### Prerequisites

- Python 3.11.9 or higher
- pip package manager
- Google Gemini API key (for LLM integration)

### Installation

1. **Clone or download the project**:
   ```bash
   cd DocuChat-RAG
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Set up environment variables**:
   ```bash
   # Create .env file with your Gemini API key
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Upload a PDF**: Use the file uploader in the sidebar to upload a PDF document
2. **Process document**: Click "Process Document" to analyze and index the PDF
3. **Ask questions**: Once processed, you can ask questions about the document content
4. **Get answers with citations**: The AI will provide answers with page number references using Gemini LLM
5. **Run Evaluation**: Click "Run Evaluation" in the sidebar to test RAG performance with 5 predefined questions
6. **Compare Answers**: Review ground truth vs RAG answers side-by-side for manual evaluation
7. **Mark Results**: Click " Correct" or " Wrong" for each answer to calculate accuracy
8. **Export Results**: Export evaluation results to CSV for further analysis

## Technical Details

### Models Used

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Dimension: 384
  - Fast and efficient for document embeddings
  - Good balance between performance and speed

- **LLM Model**: Google Gemini 2.5 Flash
  - High-quality answer generation
  - Fast response times
  - Supports chat history and context
  - Requires API key for access

### Chunking Strategy

- **Method**: RecursiveCharacterTextSplitter
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Separators**: `["\n\n", "\n", " ", ""]`

This strategy ensures:
- Coherent text chunks that maintain context
- Overlapping chunks to prevent information loss at boundaries
- Intelligent splitting at natural text boundaries

### Retrieval Strategy

- **Method**: Cosine similarity search
- **Top-K Results**: 4 documents per query
- **Vector Store**: ChromaDB with persistent storage
- **Embedding Dimension**: 384 (matching the model)

**Why this strategy works well:**
- Cosine similarity is effective for semantic text matching
- Top-4 results provide sufficient context without overwhelming the model
- ChromaDB offers fast, scalable vector storage with persistence

### RAG Pipeline

The application follows an enhanced RAG pipeline with Gemini LLM integration:

1. **Document Loading**: Extract text and metadata from PDF
2. **Chunking**: Split documents into manageable pieces
3. **Embedding**: Convert text chunks to vector representations
4. **Storage**: Store vectors in ChromaDB for fast retrieval
5. **Retrieval**: Find relevant chunks for user queries
6. **Generation**: Generate high-quality answers using Google Gemini 2.5 Flash with context and chat history
7. **Evaluation**: Test RAG performance with predefined questions and manual accuracy calculation

### Custom Prompt Template

```
You are a helpful assistant. Answer ONLY from the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer with citations like:
(Source: Page X)
```

This prompt ensures:
- Answers are grounded in the provided documents
- Clear citation format for verification
- Honest responses when information is not available

## Limitations

- **PDF Quality**: Works best with text-based PDFs. Scanned documents may require OCR
- **Document Size**: Very large PDFs (>1000 pages) may require additional processing time
- **Answer Quality**: Currently uses simplified answer generation. For production use, integrate with LLM APIs
- **Language**: Primarily designed for English documents
- **Memory Usage**: Large documents may require significant RAM for embeddings

## Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM and RAG utilities
- **sentence-transformers**: Text embeddings
- **chromadb**: Vector database
- **pypdf**: PDF text extraction
- **numpy**: Numerical operations
- **google-generativeai**: Google Gemini LLM integration
- **python-dotenv**: Environment variable management
- **pandas**: Data handling for evaluation system

## Troubleshooting

### Common Issues

1. **"Failed to load embedding model"**
   - Check internet connection for first-time model download
   - Ensure sufficient disk space (~500MB for model)

2. **"PDF loading failed"**
   - Verify the file is a valid PDF
   - Check if the PDF is password-protected
   - Try with a different PDF file

3. **"ChromaDB initialization failed"**
   - Ensure write permissions in the project directory
   - Check available disk space

4. **"GEMINI_API_KEY not found"**
   - Ensure .env file exists in project root
   - Verify API key is correctly formatted: `GEMINI_API_KEY=your_key_here`
   - Check that .env file is not in .gitignore

5. **Memory issues with large PDFs**
   - Try reducing chunk size in `rag/chunking.py`
   - Close other applications to free up RAM

6. **"Evaluation results not showing"**
   - Ensure document is processed before running evaluation
   - Check that evaluation completed successfully
   - Refresh the page if results don't appear

### Performance Tips

- Use smaller chunk sizes (500-800) for faster processing
- Clear the vector store before processing new documents
- Restart the app if it becomes slow after multiple uploads

## Future Enhancements

- Integration with LLM APIs (OpenAI, Anthropic, etc.)
- Support for multiple document formats (DOCX, TXT, etc.)
- Advanced chunking strategies
- Document comparison features
- Export chat history
- User authentication and session persistence

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests!
