import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Document Genie", layout="wide")
st.markdown("""
## Document Genie: Get instant insights from your documents ##
### How it works ###
Follow the instructions:
1) Enter your API key
2) Upload your documents
3) Ask a question
""")

api_key = st.text_input("Enter your Google API key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as accurately as possible based on the provided context.
    If the answer is not in the document, state that explicitly.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, api_key):
    if not os.path.exists("faiss_index"):
        st.error("No documents have been processed yet. Please upload and process a document first.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response.get("output_text", "No response generated."))

# Streamlit UI
def main():
    st.header("Document Genie")
    user_question = st.text_input("Ask a question from PDF files", key="user_question")
    
    if user_question and api_key:
        user_input(user_question, api_key)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your docs", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text extracted from the uploaded documents. Please try another file.")
                    return
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Processing completed successfully!")

if __name__ == "__main__":
    main()