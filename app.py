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
import pandas as pd
from io import BytesIO

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
        gemini_rag = GeminiRAG(
            api_key=api_key,
            model_name="gemini-2.5-flash",
            summary_mode="transformer",
        )
        
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


def parse_fixed_entities(ner_text: str) -> dict:
    """Parse fixed-category NER output (PERSON/LOCATION/ORGANIZATION/DATE/OTHER)."""
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

        matched_header = False
        for category in entities.keys():
            if line.upper().startswith(category + ':'):
                current_category = category
                items = line.split(':', 1)[1].strip()
                if items:
                    entities[category].extend([item.strip() for item in items.split(',') if item.strip()])
                matched_header = True
                break

        if matched_header:
            continue

        if current_category and not line.startswith('-'):
            clean_line = line.lstrip('- ').strip()
            if clean_line:
                entities[current_category].append(clean_line)

    return entities


def generate_response_for_mode(gemini_rag: GeminiRAG, mode: str, query: str, context_chunks):
    """Generate response while temporarily switching summary mode."""
    original_mode = gemini_rag.summary_mode
    try:
        gemini_rag.summary_mode = mode
        return gemini_rag.answer_question(query=query, context_chunks=context_chunks)
    finally:
        gemini_rag.summary_mode = original_mode


def export_to_excel(response, query="", response_type="Answer"):
    """Export response data to Excel format"""
    # Parse named entities using fixed categories
    ner_text = response["named_entities"]
    entities = parse_fixed_entities(ner_text)
    
    # Create Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Type': ['Query', 'Tamil Summary', 'English Summary'],
            'Content': [
                query,
                response['tamil_summary'],
                response['english_summary']
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Named Entities sheet
        ner_data = []
        for category, items in entities.items():
            if items:
                for item in items:
                    ner_data.append({'Category': category, 'Entity': item})
            else:
                ner_data.append({'Category': category, 'Entity': 'None found'})
        
        df_ner = pd.DataFrame(ner_data)
        df_ner.to_excel(writer, sheet_name='Named Entities', index=False)
        
        # Metadata sheet
        metadata = {
            'Field': ['Response Type', 'Generated At', 'System'],
            'Value': [
                response_type,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Tamil PDF QA System'
            ]
        }
        df_metadata = pd.DataFrame(metadata)
        df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    output.seek(0)
    return output


def display_response(response, show_export=False, query="", response_type="Answer"):
    """Display structured response with fixed-category NER layout"""
    
    # Tamil Summary
    st.markdown("### 📝 சுருக்கம் (Tamil Summary)")
    st.markdown(f'<div class="summary-box tamil-text">{response["tamil_summary"]}</div>', 
                unsafe_allow_html=True)
    
    # English Summary
    st.markdown("### 🌍 English Summary")
    st.markdown(f'<div class="summary-box">{response["english_summary"]}</div>', 
                unsafe_allow_html=True)
    
    # Named Entities
    st.markdown("### 🏷️ Named Entities")
    
    ner_text = response["named_entities"]
    
    entities = parse_fixed_entities(ner_text)

    col1, col2 = st.columns(2)

    with col1:
        if entities["PERSON"]:
            st.markdown("**👤 PERSON**")
            for person in entities["PERSON"]:
                st.markdown(f"• {person}")
        else:
            st.markdown("**👤 PERSON**")
            st.markdown("*None found*")

        st.markdown("")

        if entities["LOCATION"]:
            st.markdown("**📍 LOCATION**")
            for location in entities["LOCATION"]:
                st.markdown(f"• {location}")
        else:
            st.markdown("**📍 LOCATION**")
            st.markdown("*None found*")

        st.markdown("")

        if entities["DATE"]:
            st.markdown("**📅 DATE**")
            for date in entities["DATE"]:
                st.markdown(f"• {date}")
        else:
            st.markdown("**📅 DATE**")
            st.markdown("*None found*")

    with col2:
        if entities["ORGANIZATION"]:
            st.markdown("**🏢 ORGANIZATION**")
            for org in entities["ORGANIZATION"]:
                st.markdown(f"• {org}")
        else:
            st.markdown("**🏢 ORGANIZATION**")
            st.markdown("*None found*")

        st.markdown("")

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
    
    with st.expander("🐛 Debug: Raw NER Text"):
        st.code(ner_text)


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
        st.subheader("🧪 Response Mode")
        response_mode = st.radio(
            "Choose summary backend",
            ["Transformer", "Gemini", "Compare Both"],
            index=2,
            help="Compare Both shows transformer and Gemini responses side-by-side."
        )
        
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
                if response_mode == "Compare Both":
                    with st.spinner("✨ Generating Transformer and Gemini responses..."):
                        transformer_response = generate_response_for_mode(
                            gemini_rag,
                            mode="transformer",
                            query=user_query,
                            context_chunks=search_results['documents'],
                        )
                        gemini_response = generate_response_for_mode(
                            gemini_rag,
                            mode="gemini",
                            query=user_query,
                            context_chunks=search_results['documents'],
                        )

                    st.markdown("## 📋 Response Comparison")
                    tab1, tab2 = st.tabs(["Transformer Summary", "Gemini Summary"])

                    with tab1:
                        display_response(transformer_response, query=user_query, response_type="Q&A Answer - Transformer")
                        transformer_excel = export_to_excel(
                            transformer_response,
                            query=user_query,
                            response_type="Q&A Answer - Transformer"
                        )
                        st.download_button(
                            label="📥 Download Transformer Response",
                            data=transformer_excel,
                            file_name=f"qa_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_qa_transformer"
                        )

                    with tab2:
                        display_response(gemini_response, query=user_query, response_type="Q&A Answer - Gemini")
                        gemini_excel = export_to_excel(
                            gemini_response,
                            query=user_query,
                            response_type="Q&A Answer - Gemini"
                        )
                        st.download_button(
                            label="📥 Download Gemini Response",
                            data=gemini_excel,
                            file_name=f"qa_gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_qa_gemini"
                        )
                else:
                    selected_mode = "transformer" if response_mode == "Transformer" else "gemini"
                    with st.spinner(f"✨ Generating {response_mode} response..."):
                        response = generate_response_for_mode(
                            gemini_rag,
                            mode=selected_mode,
                            query=user_query,
                            context_chunks=search_results['documents'],
                        )

                    st.markdown(f"## 📋 {response_mode} Response")
                    display_response(response, show_export=True, query=user_query, response_type=f"Q&A Answer - {response_mode}")

                    excel_file = export_to_excel(response, query=user_query, response_type=f"Q&A Answer - {response_mode}")
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_file,
                        file_name=f"qa_response_{response_mode.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_qa"
                    )
    
    # Document summary feature
    st.markdown("---")
    if st.button("📄 Generate Document Summary"):
        with st.spinner("Generating summary of the entire document..."):
            # Get some chunks for summary
            all_chunks = vector_store.collection.get()
            
            if all_chunks and all_chunks['documents']:
                chunks_for_summary = all_chunks['documents'][:10]

                summary_query = "இந்த ஆவணத்தின் முக்கிய உள்ளடக்கம் என்ன? (What is the main content of this document?)"

                if response_mode == "Compare Both":
                    transformer_summary = generate_response_for_mode(
                        gemini_rag,
                        mode="transformer",
                        query=summary_query,
                        context_chunks=chunks_for_summary,
                    )
                    gemini_summary = generate_response_for_mode(
                        gemini_rag,
                        mode="gemini",
                        query=summary_query,
                        context_chunks=chunks_for_summary,
                    )

                    st.markdown("## 📄 Document Summary Comparison")
                    tab1, tab2 = st.tabs(["Transformer Summary", "Gemini Summary"])

                    with tab1:
                        display_response(transformer_summary, query="Document Summary", response_type="Full Summary - Transformer")
                    with tab2:
                        display_response(gemini_summary, query="Document Summary", response_type="Full Summary - Gemini")
                else:
                    selected_mode = "transformer" if response_mode == "Transformer" else "gemini"
                    summary_response = generate_response_for_mode(
                        gemini_rag,
                        mode=selected_mode,
                        query=summary_query,
                        context_chunks=chunks_for_summary,
                    )

                    st.markdown(f"## 📄 Document Summary ({response_mode})")
                    display_response(summary_response, show_export=True, query="Document Summary", response_type=f"Full Document Summary - {response_mode}")

                    excel_file = export_to_excel(summary_response, query="Document Summary", response_type=f"Full Document Summary - {response_mode}")
                    st.download_button(
                        label="📥 Download Summary as Excel",
                        data=excel_file,
                        file_name=f"document_summary_{response_mode.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_summary"
                    )
            else:
                st.warning("No documents in database")


if __name__ == "__main__":
    main()
