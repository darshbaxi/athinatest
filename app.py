import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import random
import pandas as pd
import json
# Configure the Google API key
GoogleAPIKey = st.secrets["GOOGLEAPIKEY"]
genai.configure(api_key=GoogleAPIKey)

# Define the Generative Model
genai_model = genai.GenerativeModel('gemini-pro')








def question_answer(context, question):
    qa_prompt = f'''
    {{
        "name": "answer_formulate",
        "instruction": "Answer the question using the information from the given context. Output verdict as '1' if answer is present '-1' if answer is not present in the context.",
        "examples": [
            {{
                "context": "Climate change is significantly influenced by human activities, notably the emission of greenhouse gases from burning fossil fuels. The increased greenhouse gas concentration in the atmosphere traps more heat, leading to global warming and changes in weather patterns.",
                "question": "How do human activities contribute to climate change?",
                "answer": {{
                    "answer": "Human activities contribute to climate change primarily through the emission of greenhouse gases from burning fossil fuels. These emissions increase the concentration of greenhouse gases in the atmosphere, which traps more heat and leads to global warming and altered weather patterns.",
                    "verdict": "1"
                }}
            }},
            {{
                "context": "The concept of artificial intelligence (AI) has evolved over time, but it fundamentally refers to machines designed to mimic human cognitive functions. AI can learn, reason, perceive, and, in some instances, react like humans, making it pivotal in fields ranging from healthcare to autonomous vehicles.",
                "question": "What are the key capabilities of artificial intelligence?",
                "answer": {{
                    "answer": "Artificial intelligence is designed to mimic human cognitive functions, with key capabilities including learning, reasoning, perception, and reacting to the environment in a manner similar to humans. These capabilities make AI pivotal in various fields, including healthcare and autonomous driving.",
                    "verdict": "1"
                }}
            }},
            {{
                "context": "The novel 'Pride and Prejudice' by Jane Austen revolves around the character Elizabeth Bennet and her family. The story is set in the 19th century in rural England and deals with issues of marriage, morality, and misconceptions.",
                "question": "What year was 'Pride and Prejudice' published?",
                "answer": {{
                    "answer": "The answer to given question is not present in context",
                    "verdict": "-1"
                }}
            }}
        ],
        "input_keys": ["{context}", "{question}"],
        "output_key": "answer",
        "output_type": "json",
        "language": "english"
    }}
    '''
    response = genai_model.generate_content(qa_prompt)
    print(response)
    candidates = response.candidates
    content_parts = candidates[0].content.parts
    text = content_parts[0].text
    text=json.loads(text)
    text=text['answer']['answer']
    return text


print(question_answer("The novel 'Pride and Prejudice' by Jane Austen revolves around the character Elizabeth Bennet and her family. The story is set in the 19th century in rural England and deals with issues of marriage, morality, and misconceptions.","What are the key capabilities of artificial intelligence?"))