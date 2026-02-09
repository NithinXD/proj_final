"""
Tamil PDF QA System - Streamlit UI
RAG-based Question Answering for Tamil Documents using Google Gemini
"""
import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from gemini_rag import GeminiRAG
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Tamil PDF QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tamil-text {
        font-size: 1.1rem;
        line-height: 1.8;
        font-family: 'Noto Sans Tamil', sans-serif;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #000000;
    }
    .entity-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #000000;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the RAG system components"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found! Please set it in .env file")
        return None, None, None
    
    try:
        pdf_processor = PDFProcessor(chunk_size=400, chunk_overlap=100)
        vector_store = VectorStore(
            collection_name="tamil_docs",
            persist_directory="./chroma_db",
            api_key=api_key  # Pass API key for Gemini embeddings
        )
        gemini_rag = GeminiRAG(api_key=api_key, model_name="gemini-2.5-flash")
        
        return pdf_processor, vector_store, gemini_rag
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None, None


def process_uploaded_pdf(uploaded_file, pdf_processor, vector_store):
    """Process and index uploaded PDF"""
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process PDF
        with st.spinner("📄 Processing PDF..."):
            result = pdf_processor.process_pdf(temp_path)
        
        st.success(f"✅ Extracted {result['num_chunks']} chunks ({result['total_chars']} characters)")
        
        # Add to vector store
        with st.spinner("🔍 Creating embeddings and indexing..."):
            doc_id = f"doc_{int(time.time())}"
            metadata = [
                {
                    "chunk_id": i, 
                    "doc_id": doc_id,
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now().isoformat()
                } 
                for i in range(len(result['chunks']))
            ]
            vector_store.add_documents(
                chunks=result['chunks'],
                metadata=metadata,
                doc_id=doc_id
            )
        
        st.success(f"✅ Document indexed! Total chunks in database: {vector_store.get_collection_count()}")
        
        return result, doc_id
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def display_response(response):
    """Display the structured response with improved NER formatting"""
    
    # Tamil Summary
    st.markdown("### 📝 சுருக்கம் (Tamil Summary)")
    st.markdown(f'<div class="summary-box tamil-text">{response["tamil_summary"]}</div>', 
                unsafe_allow_html=True)
    
    # English Summary
    st.markdown("### 🌍 English Summary")
    st.markdown(f'<div class="summary-box">{response["english_summary"]}</div>', 
                unsafe_allow_html=True)
    
    # Named Entities - Parse and format nicely
    st.markdown("### 🏷️ Named Entities")
    
    ner_text = response["named_entities"]
    
    # Parse the NER output
    entities = {
        "PERSON": [],
        "LOCATION": [],
        "ORGANIZATION": [],
        "DATE": [],
        "OTHER": []
    }
    
    current_category = None
    for line in ner_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a category header
        for category in entities.keys():
            if line.startswith(category + ':'):
                current_category = category
                # Extract items after colon
                items = line.split(':', 1)[1].strip()
                if items:
                    # Split by comma and clean
                    items_list = [item.strip() for item in items.split(',') if item.strip()]
                    entities[category].extend(items_list)
                break
        else:
            # Not a category header, add to current category if exists
            if current_category and line and not line.startswith('-'):
                # Remove leading dash if present
                clean_line = line.lstrip('- ').strip()
                if clean_line:
                    entities[current_category].append(clean_line)
    
    # Display entities in organized columns
    col1, col2 = st.columns(2)
    
    with col1:
        # PERSON
        if entities["PERSON"]:
            st.markdown("**👤 PERSON**")
            for person in entities["PERSON"]:
                st.markdown(f"• {person}")
        else:
            st.markdown("**👤 PERSON**")
            st.markdown("*None found*")
        
        st.markdown("")
        
        # LOCATION
        if entities["LOCATION"]:
            st.markdown("**📍 LOCATION**")
            for location in entities["LOCATION"]:
                st.markdown(f"• {location}")
        else:
            st.markdown("**📍 LOCATION**")
            st.markdown("*None found*")
        
        st.markdown("")
        
        # DATE
        if entities["DATE"]:
            st.markdown("**📅 DATE**")
            for date in entities["DATE"]:
                st.markdown(f"• {date}")
        else:
            st.markdown("**📅 DATE**")
            st.markdown("*None found*")
    
    with col2:
        # ORGANIZATION
        if entities["ORGANIZATION"]:
            st.markdown("**🏢 ORGANIZATION**")
            for org in entities["ORGANIZATION"]:
                st.markdown(f"• {org}")
        else:
            st.markdown("**🏢 ORGANIZATION**")
            st.markdown("*None found*")
        
        st.markdown("")
        
        # OTHER
        if entities["OTHER"]:
            st.markdown("**📚 OTHER**")
            for other in entities["OTHER"]:
                st.markdown(f"• {other}")
        else:
            st.markdown("**📚 OTHER**")
            st.markdown("*None found*")
    
    # Show raw response in expander
    with st.expander("🔍 View Raw Response"):
        st.text(response.get("raw_response", ""))


def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Tamil PDF Question-Answering System</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.1rem; color: #666;">
    Retrieval-Augmented Generation (RAG) with Google Gemini<br>
    தமிழ் PDF ஆவணங்களுக்கான கேள்வி-பதில் அமைப்பு
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize system
    pdf_processor, vector_store, gemini_rag = initialize_system()
    
    if not all([pdf_processor, vector_store, gemini_rag]):
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Upload PDF
        st.subheader("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a Tamil PDF file",
            type=['pdf'],
            help="Upload a PDF document in Tamil or bilingual Tamil-English"
        )
        
        if uploaded_file:
            if st.button("🔄 Process PDF"):
                result, doc_id = process_uploaded_pdf(uploaded_file, pdf_processor, vector_store)
                if result:
                    st.session_state['last_doc_id'] = doc_id
                    st.session_state['last_result'] = result
        
        st.markdown("---")
        
        # Database info
        st.subheader("📊 Database Stats")
        chunk_count = vector_store.get_collection_count()
        st.metric("Total Chunks", chunk_count)
        
        # Clear database
        if st.button("🗑️ Clear Database"):
            if st.confirm("Are you sure?"):
                vector_store.clear_collection()
                st.success("Database cleared!")
                st.rerun()
        
        st.markdown("---")
        
        # Retrieval settings
        st.subheader("🎛️ Retrieval Settings")
        top_k = st.slider("Number of chunks to retrieve", 3, 10, 5)
        
        st.markdown("---")
        st.markdown("""
        ### 📖 How to use:
        1. Upload a Tamil PDF
        2. Wait for processing
        3. Ask questions in Tamil or English
        4. Get structured answers with:
           - Tamil summary
           - English translation
           - Named entities
        """)
    
    # Main area - Q&A Interface
    st.header("💬 Ask Questions")
    
    # Check if database has documents
    if vector_store.get_collection_count() == 0:
        st.info("👈 Please upload a PDF document to begin")
        st.stop()
    
    # Question input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter your question (Tamil or English):",
            placeholder="e.g., இந்த ஆவணம் எதைப் பற்றியது? or What is this document about?",
            key="user_query"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", type="primary")
    
    # Process query
    if ask_button and user_query:
        with st.spinner("🤔 Thinking..."):
            # Retrieve relevant chunks
            search_results = vector_store.similarity_search(user_query, k=top_k)
            
            if not search_results['documents']:
                st.warning("No relevant information found in the database")
            else:
                # Show retrieved chunks
                with st.expander(f"📚 Retrieved {len(search_results['documents'])} relevant chunks"):
                    for i, (doc, dist) in enumerate(zip(search_results['documents'], 
                                                        search_results['distances'])):
                        st.markdown(f"**Chunk {i+1}** (similarity: {1-dist:.3f})")
                        st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                        st.markdown("---")
                
                # Generate answer
                with st.spinner("✨ Generating structured response..."):
                    response = gemini_rag.answer_question(
                        query=user_query,
                        context_chunks=search_results['documents']
                    )
                
                # Display response
                st.markdown("## 📋 Response")
                display_response(response)
    
    # Document summary feature
    st.markdown("---")
    if st.button("📄 Generate Document Summary"):
        with st.spinner("Generating summary of the entire document..."):
            # Get some chunks for summary
            all_chunks = vector_store.collection.get()
            
            if all_chunks and all_chunks['documents']:
                chunks_for_summary = all_chunks['documents'][:10]
                
                summary_response = gemini_rag.summarize_document(chunks_for_summary)
                
                st.markdown("## 📄 Document Summary")
                display_response(summary_response)
            else:
                st.warning("No documents in database")


if __name__ == "__main__":
    main()
