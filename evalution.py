from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def statement_generation(question, answer, sentences):
    prompt = f'''
    {{
        "instruction": "Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.",
        "examples": [
            {{
                ""question": "Who was Albert Einstein and what is he best known for?",
                "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
                "sentences": """
                            0:He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. 
                            1:He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
                            """,
                ],
                "analysis": [
                    {{
                        "sentence_index": 0,
                        "simpler_statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time."
                        ]
                    }},
                    {{
                        "sentence_index": 1,
                        "simpler_statements": [
                            "Albert Einstein was best known for developing the theory of relativity.",
                            "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."
                        ]
                    }}
                ]
            }}
        ],
        "input_keys": ["{question}", "{answer}", "{sentences}"],
        "output_key": "analysis",
        "language": "english"
    }}
    '''
    simpler_statements = []
    try:
        response = genai_model.generate_content(prompt)
        print(response)
        candidates = response.candidates
        content_parts = candidates[0].content.parts
        text = content_parts[0].text
        print(text)
        text = json.loads(text)
        text = text['analysis']
        for analysis in text:
            simpler_statements.extend(analysis["simpler_statements"])
    except Exception as e:
        print(f"Error during statement generation: {e}")
        return []

    return simpler_statements


def _create_statements_prompt(generated_answer, query):
    text = generated_answer
    sentences = [
        sentence for sentence in text if sentence.strip().endswith(".")
    ]

    return statement_generation(query, generated_answer, sentences)


def verdict_cnt(context, statements):
    verdict_cnt_Prompt = f'''
    {{
    instruction="Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.",
    examples=[
        {{
            "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
            "statements": [
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
            "answer": 
                [
                    {{
                        "statement": "John is majoring in Biology.",
                        "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        "verdict": 0,
                    }},
                    {{
                        "statement": "John is taking a course on Artificial Intelligence.",
                        "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        "verdict": 0,
                    }},
                    {{
                        "statement": "John is a dedicated student.",
                        "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        "verdict": 1,
                    }},
                    {{
                        "statement": "John has a part-time job.",
                        "reason": "There is no information given in the context about John having a part-time job.",
                        "verdict": 0,
                    }},
                ]
        
        }},
        {{
            "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
            "statements": ["Albert Einstein was a genius."],
            "answer": 
                [
                    {{
                        "statement": "Albert Einstein was a genius.",
                        "reason": "The context and statement are unrelated",
                        "verdict": 0,
                    }}
                ]
        }},
    ],
    input_keys=["{context}", "{statements}"],
    output_key="answer",
    output_type="json",
    language="english",
    }}
    '''

    try:
        response = genai_model.generate_content(verdict_cnt_Prompt)
        candidates = response.candidates
        content_parts = candidates[0].content.parts
        text = content_parts[0].text
        text = json.loads(text)
        answer = text['answer']
    except Exception as e:
        print(f"Error during verdict count: {e}")
        return []

    return answer


def faithfulness(context, generated_answer, query):
    try:
        simpler_statements = _create_statements_prompt(generated_answer, query)
        if not simpler_statements:
            return 0.0
        
        verdict_cnt_answer = verdict_cnt(context, simpler_statements)
        if not verdict_cnt_answer:
            return 0.0
        
        verct = 0
        for verdict in verdict_cnt_answer:
            if verdict['verdict'] == 1:
                verct += 1
        return verct / len(verdict_cnt_answer)
    except Exception as e:
        print(f"Error during faithfulness calculation: {e}")
        return 0.0

# Example usage:
# print(faithfulness('He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.','Who was Albert Einstein and what is he best known for?'))
