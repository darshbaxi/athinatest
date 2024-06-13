import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from pinecone import Pinecone
import google.generativeai as genai
from langchain.docstore.document import Document

class PDFChatbot:
    def __init__(self, google_api_key, pinecone_api_key):
        self.googleAPIKey = google_api_key
        self.pineconeAPIKey = pinecone_api_key
        self.genai_model = None
        self.index = None
        self.instructor_embeddings = None
        self.initialize()

    def initialize(self):
        # Configure Google Generative AI
        genai.configure(api_key=self.googleAPIKey)
        self.genai_model = genai.GenerativeModel('gemini-pro')

        # Configure Pinecone
        self.indexName = "pdfchatbot"
        self.index = Pinecone(api_key=self.pineconeAPIKey).Index(self.indexName)

        # Initialize the embedding model
        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def store_embeddings(self, chunks):
        for i, chunk in enumerate(chunks):
            embedding = self.instructor_embeddings.embed_query(chunk)
            self.index.upsert(vectors=[{
                'id': str(i),
                'values': embedding,
                'metadata': {'text': chunk}
            }])

    def reply(self, question):
        e = self.instructor_embeddings.embed_query(question)
        data = self.index.query(vector=e, top_k=2, include_values=True)
        matches = data['matches']

        prompts_responses = []
        for match in matches[:2]:
            extracted_id = int(match['id'])

            try:
                fetch = self.index.fetch([str(extracted_id)]) 
                context = fetch['vectors'][str(extracted_id)]['metadata']['text']
                if context:
                    prompts_responses.append(context)
            except Exception as e:
                st.error(f"Error: {e}")

        context = " ".join(prompts_responses)
        response = self.get_conversational_chain(context, question)
        return self.extract_text_from_response(response), context

    def get_conversational_chain(self, context, question):
        prompt_template = f"""
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. Make sure that Answer is about of 200 words.
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        response = self.genai_model.generate_content(prompt_template)
        return response

    def extract_text_from_response(self, response):
        try:
            candidates = response.candidates
            content_parts = candidates[0].content.parts
            text = content_parts[0].text
            return text
        except (AttributeError, IndexError) as e:
            st.error(f"Error extracting text: {e}")
            return None
