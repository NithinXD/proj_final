# Tamil PDF Question-Answering System

A Retrieval-Augmented Generation (RAG) pipeline for Tamil PDF documents using Google Gemini.

## 🌟 Features

- **Tamil PDF Processing**: Extract and process Tamil Unicode text from PDFs
- **Intelligent Chunking**: Semantic chunking with paragraph/sentence boundaries
- **Vector Search**: Multilingual embeddings with ChromaDB
- **RAG with Gemini**: Context-grounded responses using Google Gemini 1.5
- **Structured Output**: Three-block response format:
  - தமிழ் சுருக்கம் (Tamil Summary)
  - English Summary
  - Named Entity Recognition
- **Interactive UI**: Streamlit-based web interface

## 📋 Requirements

- Python 3.8+
- Google Gemini API Key
- 4GB RAM minimum (for embedding model)

## 🚀 Installation

1. **Clone or download this project**

2. **Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up API key**:
   - Copy `.env.example` to `.env`
   - Add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## 🎯 Usage

### Start the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the system:

1. **Upload PDF**: Click "Browse files" in the sidebar and select a Tamil PDF
2. **Process**: Click "Process PDF" to extract and index the document
3. **Ask Questions**: Enter questions in Tamil or English in the main area
4. **Get Answers**: Receive structured responses with:
   - Tamil summary
   - English translation
   - Named entities (persons, locations, organizations, dates)

### Example Questions:

**Tamil:**
- இந்த ஆவணத்தின் முக்கிய கருத்து என்ன?
- முக்கியமான நபர்கள் யார்?
- எந்த இடங்கள் குறிப்பிடப்பட்டுள்ளன?

**English:**
- What is the main topic of this document?
- Who are the key persons mentioned?
- What are the important dates?

## 📂 Project Structure

```
proj_final/
├── app.py                 # Streamlit UI application
├── pdf_processor.py       # PDF text extraction and chunking
├── vector_store.py        # ChromaDB vector database management
├── gemini_rag.py          # RAG pipeline with Gemini
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in git)
├── .env.example           # Template for .env
├── .gitignore            # Git ignore file
├── README.md             # This file
└── chroma_db/            # Vector database (auto-created)
```

## 🔧 Configuration

### Adjust retrieval settings:
- **Top-k**: Number of chunks to retrieve (default: 5)
- **Chunk size**: Token size per chunk (default: 400)
- **Chunk overlap**: Overlap between chunks (default: 100)

Edit these in the sidebar of the Streamlit app or modify defaults in source files.

### Change Gemini model:
In `app.py`, modify:
```python
gemini_rag = GeminiRAG(api_key=api_key, model_name="gemini-1.5-pro")
```
- `gemini-1.5-flash`: Faster, lower cost
- `gemini-1.5-pro`: Higher quality, slower

## 🧪 Testing Individual Modules

### Test PDF processor:
```bash
python pdf_processor.py
```

### Test vector store:
```bash
python vector_store.py
```

### Test Gemini RAG:
```bash
python gemini_rag.py
```

## 🐛 Troubleshooting

### Error: "GOOGLE_API_KEY not found"
- Ensure `.env` file exists with valid API key
- Check that `python-dotenv` is installed

### Error: Embedding model download fails
- Check internet connection
- Model downloads ~1GB on first run
- Alternative: Use smaller model in `vector_store.py`

### Tamil text not displaying correctly
- Install Tamil Unicode fonts
- Use Chrome/Firefox for best Tamil support

### PDF extraction returns empty text
- Ensure PDF has selectable text (not scanned image)
- Try different PDF processing library (toggle in `pdf_processor.py`)

## 📊 Performance Notes

- **First run**: ~5-10 minutes (downloads embedding model ~1GB)
- **PDF processing**: ~5-30 seconds (depends on PDF size)
- **Query response**: ~3-8 seconds
- **Memory usage**: ~2-4GB with model loaded

## 🔐 Security

- **Never commit `.env` file** to version control
- Keep your API key secure
- Use environment-specific API keys for production

## 📖 Research Paper

This implementation is based on the research paper:
**"Multilingual Document Question-Answering System for Tamil PDFs Using Retrieval-Augmented Generation with Google Gemini"**

See `grok_report.pdf` for full technical details.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for scanned PDFs (OCR)
- Multi-document search
- Chat history
- Citation highlighting
- Support for other Indic languages

## 📄 License

MIT License - feel free to use for academic or commercial projects.

## 👨‍💻 Author

Developed as part of Tamil NLP research initiative.

## 🙏 Acknowledgments

- Google Gemini API
- Sentence Transformers (multilingual embeddings)
- ChromaDB team
- Tamil computing community
