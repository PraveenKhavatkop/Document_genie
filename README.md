# Document_genie

## Introduction:-

**- Problem Statement:**
Retrieving relevant patent information manually is time-consuming.
**- Solution:**
AI-powered chatbot for instant patent insights.

## Technology Stack
- **Programming Language:** Python
- **Framework:** Streamlit
- **Libraries & Tools:**
LangChain
FAISS
Google Gemini API
PyPDF2

## System Architecture
- User uploads patents PDF & asks queries
- Extract text from PDFs
- Convert text into embeddings using FAISS
- Retrieve relevant text & generate AI responses
- Display results on Streamlit UI

## Implementation Steps
1. Extract text using PyPDF2
2. Chunk text for better processing
3. Convert chunks to vector embeddings
4. Store in FAISS index
5. Retrieve & generate answers using Gemini AI

## Demo Workflow
- Step 1: Enter API Key
- Step 2: Upload Patent PDFs
- Step 3: Ask a Question
- Step 4: Retrieve and Display Answers

## Key Achievements
- Successfully implemented chatbot
- Real-time document search
- High accuracy in patent-related queries
- Scalable & extendable for multiple domains

## Future Enhancement
- Improve accuracy with RAG (Retrieval-Augmented Generation)
- Support more file formats (Word, Excel, etc.)
- Expand patent database integration
- Deploy as a cloud-based API

## Conclusion
- AI-driven patent intelligence
- Reduces manual research effort
- Enhancements will make it a robust legal & R&D tool

