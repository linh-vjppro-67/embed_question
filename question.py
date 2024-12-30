import streamlit as st
import openai
import numpy as np
import json
import faiss
import os

# Configure Azure OpenAI from secrets
openai.api_type = "azure"
openai.api_key = st.secrets["azure_openai"]["api_key"]  # Retrieve API Key from Streamlit Secrets
openai.api_base = st.secrets["azure_openai"]["api_base"]  # Retrieve endpoint from Streamlit Secrets
openai.api_version = "2023-05-15"  # Use the latest API version

# Function to create embeddings for a query (old API)
def get_query_embedding(query):
    try:
        response = openai.Embedding.create(  # Use openai.Embedding.create
            engine="text-embedding-3-large", 
            input=query
        )
        return np.array(response['data'][0]['embedding'])
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Load JSON file containing skill and category data
@st.cache_data
def load_skill_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load JSON file containing questions and embeddings
@st.cache_data
def load_question_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for item in data:
        item['embedding'] = np.array(item['embedding'])
    return data

# Streamlit Interface
st.title("Search for Questions by Category and Skill")
st.write("Select a category, skill(s), and enter your query to perform a search.")

# Path to JSON files
skill_data_file = './data.json'
question_data_file = './combined_data.json'

# Load data
skill_data = load_skill_data(skill_data_file)
question_data = load_question_data(question_data_file)

# Create category selection
categories = [category['category'] for category in skill_data['value']]
selected_category = st.selectbox("Select Category:", options=categories)

# Select skills (multiple)
skills = []
for category in skill_data['value']:
    if category['category'] == selected_category:
        skills = category['skills']
        break
selected_skills = st.multiselect("Select Skills:", options=skills)

# Automatically create a query based on the selection
query = f"Search for 5 questions with category '{selected_category}' and skills {', '.join(selected_skills)} about:"

# Display the query in the input box
query_input = st.text_input("Enter your query:", value=query)

# Create FAISS index from question data
def create_faiss_index(data):
    embeddings = np.array([item['embedding'] for item in data])
    dimension = embeddings.shape[1]  # Vector dimension
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index with L2 (Euclidean) distance
    index.add(embeddings)  # Add embeddings to the index
    return index

# Find similar questions using FAISS
def find_top_questions_faiss(query_embedding, index, data, top_n=5):
    distances, indices = index.search(np.array([query_embedding]), top_n)  # Find the most similar questions
    top_questions = [data[idx] for idx in indices[0]]  # Retrieve questions by indices
    return top_questions

# Filter questions by category and skill
def filter_questions_by_category_and_skill(questions, category, skills):
    filtered_questions = []
    for question in questions:
        # Check filter conditions for category and skills
        if category in question.get('category', []) and any(skill in question.get('skills', []) for skill in skills):
            filtered_questions.append(question)
    return filtered_questions

# Perform search
if st.button("Search"):
    if not query_input.strip():
        st.warning("Please enter a query to perform the search.")
    else:
        st.write(f"Processing query: **{query_input}**")
        
        # Filter questions by category and skills
        filtered_questions = filter_questions_by_category_and_skill(question_data, selected_category, selected_skills)
        
        if filtered_questions:
            # Create a FAISS index from the filtered questions
            faiss_index = create_faiss_index(filtered_questions)
            
            query_embedding = get_query_embedding(query_input)
            if query_embedding is not None:
                top_questions = find_top_questions_faiss(query_embedding, faiss_index, filtered_questions)
                
                if top_questions:
                    st.write("### Top 5 most similar questions:")
                    for idx, item in enumerate(top_questions, 1):
                        st.write(f"**Question {idx}:** {item['question']}")
                        st.write(f"**Category:** {', '.join(item.get('category', []))}")
                        st.write(f"**Related Skills:** {', '.join(item.get('skills', []))}")
                        st.write("**Options:**")
                        options = item.get('options', [])
                        for option in options:
                            is_answer_key = "✅" if option.get("isAnswerKey", False) else "❌"
                            st.write(f"- {option['description']} {is_answer_key}")
                        st.write("---")
                else:
                    st.write("No similar questions found.")
            else:
                st.error("Unable to create embeddings for the query.")
        else:
            st.write("No questions found for the selected category and skills.")
