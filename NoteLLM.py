import os
import fitz  # PyMuPDF
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Function to load and extract text from a PDF file using PyMuPDF
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load the documents manually
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        
        # Use PyMuPDF to load and extract text from PDFs
        pdf_folder = "./data"  # Path to your directory with PDFs
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        # Extract text from each PDF
        st.session_state.docs = []
        for pdf_file in pdf_files:
            file_path = os.path.join(pdf_folder, pdf_file)
            text = load_pdf(file_path)
            st.session_state.docs.append(text)

        # Create embeddings for documents
        st.session_state.vectors = [st.session_state.embeddings.embed_document(doc) for doc in st.session_state.docs]

# Input for the user's question
prompt1 = input("Enter Your Question From Documents: ")

# Handling the user's query and displaying the response
if prompt1:
    vector_embedding()
    # Perform a simple similarity search (e.g., using FAISS or just a simple search)
    relevant_contexts = [doc for doc in st.session_state.docs if prompt1.lower() in doc.lower()]
    
    # Assuming you have a predefined prompt template and LLM (e.g., OpenAI)
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(input_variables=["context", "input"], template="Answer the following question based on the context: {context} Questions: {input}")
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Combine the relevant context and the user's question
    answer = chain.run(context=" ".join(relevant_contexts), input=prompt1)
    
    print(f"Answer: {answer}")
