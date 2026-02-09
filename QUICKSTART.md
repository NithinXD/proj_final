# Quick Start Guide

## Tamil PDF Question-Answering System

### Prerequisites
- Python 3.8 or higher installed
- Google Gemini API key (get from: https://ai.google.dev/)

---

## ⚡ Quick Setup (3 steps)

### Step 1: Install
Double-click `setup.bat` or run:
```bash
setup.bat
```

This will:
- Create virtual environment
- Install all dependencies (~5-10 minutes on first run)

### Step 2: Configure API Key
Open `.env` file and add your API key:
```
GOOGLE_API_KEY=your_actual_key_here
```

**Your current API key is already set in .env file!**

### Step 3: Run
Double-click `run.bat` or run:
```bash
run.bat
```

The application will open in your browser at http://localhost:8501

---

## 🎯 How to Use

### 1. Upload PDF
- Click **"Browse files"** in the left sidebar
- Select a Tamil PDF document
- Click **"Process PDF"** button

### 2. Ask Questions
In the main area:
- Type your question in Tamil or English
- Click **"Ask"** button
- Get structured response with:
  - **தமிழ் சுருக்கம்** (Tamil Summary)
  - **English Summary**
  - **Named Entities** (people, places, dates, etc.)

### 3. Sample Questions to Try

**In Tamil:**
```
இந்த ஆவணத்தின் முக்கிய கருத்து என்ன?
முக்கியமான நபர்கள் யார்?
எந்த தேதிகள் குறிப்பிடப்பட்டுள்ளன?
```

**In English:**
```
What is the main topic?
Who are the key persons mentioned?
What locations are discussed?
```

---

## 🔧 Manual Setup (if needed)

If automated setup doesn't work:

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 What Happens on First Run?

1. **Embedding Model Download** (~1GB)
   - Downloads multilingual model for Tamil support
   - Takes 5-10 minutes depending on internet speed
   - Only happens once!

2. **Database Initialization**
   - Creates `chroma_db` folder
   - Sets up vector database

---

## ⚙️ Configuration Options

Edit `config.py` to customize:

- **Model Selection:**
  ```python
  GEMINI_MODEL = "gemini-1.5-flash"  # Fast & cheap
  # or
  GEMINI_MODEL = "gemini-1.5-pro"    # Better quality
  ```

- **Chunk Settings:**
  ```python
  CHUNK_SIZE = 400        # Size of text chunks
  CHUNK_OVERLAP = 100     # Overlap between chunks
  ```

- **Retrieval Settings:**
  ```python
  DEFAULT_TOP_K = 5       # Number of chunks to retrieve
  ```

---

## 🐛 Troubleshooting

### Problem: "GOOGLE_API_KEY not found"
**Solution:** Edit `.env` file and add your API key

### Problem: Embedding model download fails
**Solution:** Check internet connection and try again

### Problem: Tamil text shows as boxes
**Solution:** Install Tamil Unicode fonts on your system

### Problem: PDF extraction is empty
**Solution:** Your PDF might be a scanned image. Try a different PDF with selectable text.

---

## 💡 Tips for Best Results

1. **Use clear, specific questions**
   - Good: "இந்த கதையில் முக்கிய பாத்திரம் யார்?"
   - Bad: "சொல்லு"

2. **Upload quality PDFs**
   - Text-based PDFs work best
   - Avoid scanned images

3. **Adjust Top-K slider**
   - More chunks = better context but slower
   - Fewer chunks = faster but might miss info

---

## 📁 Project Files

```
proj_final/
├── app.py              ← Main application
├── pdf_processor.py    ← PDF handling
├── vector_store.py     ← Vector database
├── gemini_rag.py       ← RAG with Gemini
├── config.py           ← Settings
├── .env                ← Your API key (keep secret!)
├── requirements.txt    ← Python packages
├── setup.bat           ← Setup script
└── run.bat             ← Run script
```

---

## 🚀 Next Steps

Once the system is running:

1. Try uploading a sample Tamil PDF
2. Ask various questions to test accuracy
3. Experiment with different retrieval settings
4. Check the "Document Summary" feature

---

## 📞 Need Help?

- Check README.md for detailed documentation
- Review the research paper in grok_report.pdf
- Verify all dependencies are installed correctly

---

## ✅ System Status Check

Run this to verify setup:
```bash
python config.py
```

Should show:
```
✅ Configuration is valid!
```

---

**You're all set! Enjoy using the Tamil PDF QA System! 🎉**
