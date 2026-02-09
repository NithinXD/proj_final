"""
Gemini RAG Module with LangChain Multi-Agent System
Handles RAG pipeline with specialized agents for summary, translation, and NER
"""
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from typing import List, Dict, Optional
import re


class GeminiRAG:
    """Multi-agent RAG system using LangChain and Google Gemini"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini RAG system with LangChain agents
        
        Args:
            api_key: Google API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize LangChain LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        # Initialize specialized agents
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup specialized LangChain agents for different tasks"""
        
        # Agent 1: Tamil Summary Agent
        summary_template = """You are an expert at creating concise summaries in Tamil.

Context from document:
{context}

User question: {question}

Create a natural, fluent 4-8 sentence summary in Tamil that directly answers the question based on the context.
If the context doesn't contain enough information, say so in Tamil.
Use natural Tamil - avoid literal translations.

Tamil Summary:"""
        
        self.summary_prompt = PromptTemplate(
            template=summary_template,
            input_variables=["context", "question"]
        )
        self.summary_agent = self.summary_prompt | self.llm
        
        # Agent 2: Translation Agent
        translation_template = """You are an expert Tamil to English translator.

Tamil text to translate:
{tamil_text}

Provide a faithful, natural English translation that preserves the meaning and tone.

English Translation:"""
        
        self.translation_prompt = PromptTemplate(
            template=translation_template,
            input_variables=["tamil_text"]
        )
        self.translation_agent = self.translation_prompt | self.llm
        
        # Agent 3: Named Entity Recognition Agent
        ner_template = """You are an expert at extracting named entities from Tamil and English text.

Original context from PDF:
{context}

Extract ALL named entities from the text above. Read the text carefully and identify proper nouns.

IMPORTANT INSTRUCTIONS:
1. Extract entities ONLY from the provided context
2. List each entity only ONCE (no duplicates)
3. Use the exact spelling from the text
4. Group entities by category
5. If a category has no entities, leave it empty
6. Format: One entity per line after the category label

Categories:
- PERSON: Names of people, authors, poets, historical figures
- LOCATION: Cities, countries, places, geographic locations  
- ORGANIZATION: Institutions, companies, groups, associations
- DATE: Specific dates, years, time periods, eras
- OTHER: Books, works, events, or other significant proper nouns

Output format (example):
PERSON: Name1, Name2, Name3
LOCATION: Place1, Place2
ORGANIZATION: Org1, Org2
DATE: 1882, 2020
OTHER: Book Title, Event Name

Now extract the entities:"""
        
        self.ner_prompt = PromptTemplate(
            template=ner_template,
            input_variables=["context"]
        )
        self.ner_agent = self.ner_prompt | self.llm
    
    def generate_response_multi_agent(self, query: str, context_chunks: List[str]) -> Dict[str, str]:
        """
        Generate structured response using multi-agent system
        
        Args:
            query: User question
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary with tamil_summary, english_summary, named_entities
        """
        try:
            # Combine context
            context = "\n\n---\n\n".join(context_chunks)
            
            print("🤖 Agent 1: Generating Tamil summary...")
            # Agent 1: Generate Tamil summary
            result = self.summary_agent.invoke({
                "context": context,
                "question": query
            })
            tamil_summary = result.content.strip()
            
            print("🤖 Agent 2: Translating to English...")
            # Agent 2: Translate to English
            result = self.translation_agent.invoke({
                "tamil_text": tamil_summary
            })
            english_summary = result.content.strip()
            
            print("🤖 Agent 3: Extracting named entities...")
            # Agent 3: Extract named entities from full context
            result = self.ner_agent.invoke({
                "context": context
            })
            named_entities = result.content.strip()
            
            # Format the response
            raw_response = f"""## சுருக்கம்
{tamil_summary}

## English Summary
{english_summary}

## Named Entities
{named_entities}"""
            
            return {
                'tamil_summary': tamil_summary,
                'english_summary': english_summary,
                'named_entities': named_entities,
                'raw_response': raw_response
            }
            
        except Exception as e:
            error_msg = f"Error in multi-agent generation: {str(e)}"
            print(error_msg)
            return {
                'tamil_summary': f'பிழை: {str(e)}',
                'english_summary': f'Error: {str(e)}',
                'named_entities': 'Could not extract entities',
                'raw_response': error_msg
            }
    
    def answer_question(self, query: str, context_chunks: List[str]) -> Dict[str, str]:
        """
        Complete QA pipeline using multi-agent system
        
        Args:
            query: User question in Tamil or English
            context_chunks: Retrieved relevant text chunks
            
        Returns:
            Dictionary with parsed response components
        """
        # Use multi-agent system
        return self.generate_response_multi_agent(query, context_chunks)
    
    
    def summarize_document(self, context_chunks: List[str], max_chunks: int = 10) -> Dict[str, str]:
        """
        Generate a summary of the entire document using multi-agent system
        
        Args:
            context_chunks: All document chunks (will use first max_chunks)
            max_chunks: Maximum chunks to use for summary
            
        Returns:
            Dictionary with summary components
        """
        # Use first N chunks for overview
        chunks_to_use = context_chunks[:max_chunks]
        
        query = "இந்த ஆவணத்தின் முக்கிய உள்ளடக்கம் என்ன? (What is the main content of this document?)"
        
        return self.answer_question(query, chunks_to_use)


if __name__ == "__main__":
    # Test with API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        print("Testing Gemini RAG with Multi-Agent System...")
        rag = GeminiRAG(api_key)
        print(f"Initialized with LangChain agents")
        
        # Test with sample context
        test_context = ["சுப்ரமணிய பாரதியார் தமிழ் மகாகவி. அவர் 1882ல் பிறந்தார். சென்னையில் வாழ்ந்தார்."]
        test_query = "பாரதியார் யார்?"
        
        print("\nTest query:", test_query)
        print("\nRunning multi-agent system...")
        # Uncomment to test actual API call:
        # response = rag.answer_question(test_query, test_context)
        # print("\nTamil Summary:", response['tamil_summary'])
        # print("\nEnglish Summary:", response['english_summary'])
        # print("\nNamed Entities:", response['named_entities'])
    else:
        print("GOOGLE_API_KEY not found in .env file")
