import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from pinecone import Pinecone
import google.generativeai as genai
from langchain.docstore.document import Document

# Configure Google Generative AI
GoogleAPIKey = st.secrets["GOOGLEAPIKEY"]
genai.configure(api_key=GoogleAPIKey)
genai_model = genai.GenerativeModel('gemini-pro')

# Configure Pinecone
pineconeAPIKey = st.secrets["PINECONEAPIKEY"]
pc = Pinecone(api_key=pineconeAPIKey)
indexName = "pdfchatbot"
index = pc.Index(indexName)

# Initialize the embedding model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def store_embeddings(chunks):
    for i, chunk in enumerate(chunks):
        embedding = instructor_embeddings.embed_query(chunk)
        index.upsert(vectors=[{
            'id': str(i),
            'values': embedding,
            'metadata': {'text': chunk}
        }])

def extract_text_from_response(response):
    try:
        candidates = response.candidates
        content_parts = candidates[0].content.parts
        text = content_parts[0].text
        return text
    except (AttributeError, IndexError) as e:
        st.error(f"Error extracting text: {e}")
        return None

def Reply(question):
    e = instructor_embeddings.embed_query(question)
    data = index.query(vector=e, top_k=2, include_values=True)
    matches = data['matches']

    prompts_responses = []
    for match in matches[:2]:
        extracted_id = int(match['id'])
        print("Extracted ID:", extracted_id)

        try:
            fetch = index.fetch([str(extracted_id)]) 
            context = fetch['vectors'][str(extracted_id)]['metadata']['text']
            if context:
                prompts_responses.append(context)
            else:
                print("No row")
        except Exception as e:
            st.error(f"Error: {e}")

    context = " ".join(prompts_responses)
    response = get_conversational_chain(context, question)
    return extract_text_from_response(response), context

def get_conversational_chain(context, question):
    prompt_template = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. Make sure that Answer is about of 200 words.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    response = genai_model.generate_content(prompt_template)
    print(response)
    return response



uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key='up2')

if uploaded_files:
    text = get_pdf_text(uploaded_files)
    chunks = get_text_chunks(text)
    store_embeddings(chunks)
    st.write("PDF text processed and embeddings stored successfully.")

question = st.text_input("Ask a question:")
if question:
    answer, context = Reply(question)
    st.write("Answer:", answer)
    st.write("Context:", context)


