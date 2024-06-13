from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import random
import pandas as pd
from evalution import faithfulness
import json

# Configure the Google API key
GoogleAPIKey = st.secrets["GOOGLEAPIKEY"]
genai.configure(api_key=GoogleAPIKey)

# Define the Generative Model
genai_model = genai.GenerativeModel('gemini-pro')


from langchain_helper import Reply


def reasoning_question(context, question):
    prompt = f'''
    {{
        "name": "reasoning_question",
        "instruction": """Complicate the given question by rewriting question into a multi-hop reasoning question based on the provided context. Answering the question should require the reader to make multiple logical connections or inferences using the information available in given context. 
        Rules to follow when rewriting question: 
        1. Ensure that the rewritten question can be answered entirely from the information present in the contexts. 
        2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible. 
        3. Make sure the question is clear and unambiguous. 
        4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.""",
        "examples": [
            {{
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
                "output": "Linking the Eiffel Tower and administrative center, which city stands as both?"
            }},
            {{
                "question": "What does the append() method do in Python?",
                "context": "In Python, lists are used to store multiple items in a single variable. Lists are one of 4 built-in data types used to store collections of data. The append() method adds a single item to the end of a list.",
                "output": "If a list represents a variable collection, what method extends it by one item?"
            }}
        ],
        "input_keys": ["{context}", "{question}"],
        "output_key": "output",
        "output_type": "str",
        "language": "english"
    }}
    '''
    response = genai_model.generate_content(prompt)
    candidates = response.candidates
    content_parts = candidates[0].content.parts
    text = content_parts[0].text
    return text



def conditional_question(context, question):
    prompt = f'''
    {{
        "name": "conditional_question",
        "instruction": """Rewrite the provided question to increase its complexity by introducing a conditional element.
        The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
        Follow the rules given below while rewriting the question.
        1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present context.
        4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question."""
        "examples": [
            {{
                "question": "What is the function of the roots of a plant?",
                "context": "The roots of a plant absorb water and nutrients from the soil, anchor the plant in the ground, and store food.",
                "output": "What dual purpose do plant roots serve concerning soil nutrients and stability?"
            }},
            {{
                "question": "How do vaccines protect against diseases?",
                "context": "Vaccines protect against diseases by stimulating the body's immune response to produce antibodies, which recognize and combat pathogens.",
                "output": "How do vaccines utilize the body's immune system to defend against pathogens?"
            }}
        ],
        "input_keys": ["{context}", "{question}"],
        "output_key": "output",
        "output_type": "str",
        "language": "english"
    }}
    '''
    response = genai_model.generate_content(prompt)
    candidates = response.candidates
    content_parts = candidates[0].content.parts
    text = content_parts[0].text
    return text




# Function to get text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to extract keyphrases from a given text
def keyphrase_extraction(text):
    keyphrase_extraction_prompt = f'''
    {{
        "name": "keyphrase_extraction",
        "instruction": "Extract the top 3 to 5 keyphrases from the provided text, focusing on the most significant and distinctive aspects.",
        "examples": [
            {{
                "text": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
                "output": {{
                    "keyphrases": [
                        "Black hole",
                        "Region of spacetime",
                        "Strong gravity",
                        "Light and electromagnetic waves",
                        "Theory of general relativity"
                    ]
                }}
            }},
            {{
                "text": "The Great Wall of China is an ancient series of walls and fortifications located in northern China, built around 500 years ago. This immense wall stretches over 13,000 miles and is a testament to the skill and persistence of ancient Chinese engineers.",
                "output": {{
                    "keyphrases": [
                        "Great Wall of China",
                        "Ancient fortifications",
                        "Northern China"
                    ]
                }}
            }}
        ],
        "input_keys": ["{text}"],
        "output_key": "output",
        "output_type": "json"
    }}
    '''
    response = genai_model.generate_content(keyphrase_extraction_prompt)
    candidates = response.candidates
    content_parts = candidates[0].content.parts
    text = content_parts[0].text
    return text

# Function to generate seed question
def seed_question(context, keyphrases):
    seed_question_prompt = f'''
    {{
        "name": "seed_question",
        "instruction": "Generate a question that can be fully answered from the given context. The question should be formed using the topic.",
        "examples": [
            {{
                "context": "Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.",
                "keyphrase": "Photosynthesis",
                "question": "What is the role of photosynthesis in plant growth?"
            }},
            {{
                "context": "The Industrial Revolution, starting in the 18th century, marked a major turning point in history as it led to the development of factories and urbanization.",
                "keyphrase": "Industrial Revolution",
                "question": "How did the Industrial Revolution mark a major turning point in history?"
            }},
            {{
                "context": "The process of evaporation plays a crucial role in the water cycle, converting water from liquid to vapor and allowing it to rise into the atmosphere.",
                "keyphrase": "Evaporation",
                "question": "Why is evaporation important in the water cycle?"
            }}
        ],
        "input_keys": ["{context}", "{keyphrases}"],
        "output_key": "question",
        "output_type": "str"
    }}
    '''
    response = genai_model.generate_content(seed_question_prompt)
    candidates = response.candidates
    content_parts = candidates[0].content.parts
    text = content_parts[0].text
    return text

# Function to answer the question based on context
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
    candidates = response.candidates
    content_parts = candidates[0].content.parts
    text = content_parts[0].text
    text=json.loads(text)
    text=text['answer']['answer']
    return text
    return text

# Streamlit app code
st.title("PDF Keyphrase Extraction and QA Generation")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Load and split PDF into chunks
    text = get_pdf_text([uploaded_file])
    chunks = get_text_chunks(text)    
    # Select random n chunks
    n = st.slider("Select number of random chunks", min_value=1, max_value=len(chunks))
    selected_chunks = random.sample(chunks,n)
    
    results = []
    
    for chunk in selected_chunks:
        context = chunk
        keyphrases_response = keyphrase_extraction(context)
        if not keyphrases_response:
            st.warning("No keyphrases extracted. Skipping this chunk.")
            continue
        question_response = seed_question(context, keyphrases_response)
        question_conditional=conditional_question(context,question_response)
        question_resoning=reasoning_question(context,question_response)
        answer_response = question_answer(context, question_response)
        print(question_response)
        results.append({"context": context, "query": question_response, "groundtruth": answer_response})
        results.append({"context": context, "query": question_resoning, "groundtruth": answer_response})
        results.append({"context": context, "query": question_conditional, "groundtruth": answer_response})
        
        
    # Display the results
    st.write("Generated Questions and Answers")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)
    
    
    verdict_results=[]
    if st.button('Generate testcase'):
        for result in results:
            context_llm,generated_answer=Reply(result['query'])
            score=faithfulness(result['context'],generated_answer,result['query'])
            print(score)
            verdict_results.append({"context": result['context'], "query": result['query'], "groundtruth": result['groundtruth'],"llm_answer":generated_answer,"Faithfulness":score})

        
        df=pd.DataFrame(verdict_results)
        st.dataframe(df)
        
        
    # Save results to CSV
    if st.button("Save to CSV"):
        df = pd.DataFrame(results)
        df.to_csv("results.csv", index=False)
        st.write("Results saved to CSV file")
