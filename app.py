import streamlit as st
import random
import pandas as pd
from langchain_helper import PDFChatbot 
from dataset_generation import PDFQA
from evalution import faithfulness

# Initialize Streamlit secrets and configurations
GoogleAPIKey = st.secrets["GOOGLEAPIKEY"]
PineconeAPIKey = st.secrets["PINECONEAPIKEY"]

st.title("PDF Chatbot and QA System")

# PDF Chatbot and QA System
class PDFApp:
    def __init__(self):
        self.pdf_chatbot = PDFChatbot(google_api_key=GoogleAPIKey, pinecone_api_key=PineconeAPIKey)
        self.pdf_qa = PDFQA(api_key=GoogleAPIKey)
    def pages(self):
        option=st.radio("choose evaluation or PDF chatbot",('Evaluation','PDF chatbot'))
        if option=='PDF chatbot':
            self.run()
        else:
            self.generate_qa()
    def run(self):
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key='up2')

        if uploaded_files:
            text = self.pdf_chatbot.get_pdf_text(uploaded_files)
            chunks = self.pdf_chatbot.get_text_chunks(text)
            self.pdf_chatbot.store_embeddings(chunks)
            st.write("PDF text processed and embeddings stored successfully.")

        question = st.text_input("Ask a question:")
        if question:
            answer, context = self.pdf_chatbot.reply(question)
            st.write("Answer:", answer)
            st.write("Context:", context)

        if st.button("Generate QA"):
            self.generate_qa()

    def generate_qa(self):
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file is not None:
            text = self.pdf_chatbot.get_pdf_text([uploaded_file])
            chunks = self.pdf_chatbot.get_text_chunks(text)
            
            n = st.slider("Select number of random chunks", min_value=1, max_value=len(chunks))
            selected_chunks = random.sample(chunks, n)
            
            results = []
            
            for chunk in selected_chunks:
                context = chunk
                keyphrases_response = self.pdf_qa.keyphrase_extraction(context)
                if not keyphrases_response:
                    st.warning("No keyphrases extracted. Skipping this chunk.")
                    continue
                question_response = self.pdf_qa.seed_question(context, keyphrases_response)
                question_conditional = self.pdf_qa.conditional_question(context, question_response)
                question_reasoning = self.pdf_qa.reasoning_question(context, question_response)
                answer_response = self.pdf_qa.question_answer(context, question_response)
                results.append({"context": context, "query": question_response, "groundtruth": answer_response})
                results.append({"context": context, "query": question_reasoning, "groundtruth": answer_response})
                results.append({"context": context, "query": question_conditional, "groundtruth": answer_response})
            
            st.write("Generated Questions and Answers")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            
            
            
            verdict_results = []
            if st.button('Generate testcase'):
                for result in results:
                    generated_answer, context_llm = self.reply(result['query'])
                    print(generated_answer)
                    score = faithfulness(result['context'], generated_answer, result['query'])
                    print(score)
                    verdict_results.append({"context": result['context'], "query": result['query'], "groundtruth": result['groundtruth'], "llm_answer": generated_answer, "Faithfulness": score})

                df = pd.DataFrame(verdict_results)
                st.dataframe(df)
        
    # Save results to CSV
            if st.button("Save to CSV"):
                df = pd.DataFrame(results)
                df.to_csv("results.csv", index=False)
                st.write("Results saved to CSV file")
            
            
            
            
# Run the app
app = PDFApp()
app.pages()
