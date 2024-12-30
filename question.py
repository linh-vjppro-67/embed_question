import streamlit as st
import openai
import numpy as np
import json
import faiss
import os

# Cấu hình Azure OpenAI từ secrets
openai.api_type = "azure"
openai.api_key = st.secrets["azure_openai"]["api_key"]  # Lấy API Key từ Streamlit Secrets
openai.api_base = st.secrets["azure_openai"]["api_base"]  # Lấy endpoint từ Streamlit Secrets
openai.api_version = "2023-05-15"  # Sử dụng API phiên bản mới nhất

# Hàm tạo embedding cho truy vấn (API mới)
def get_query_embedding(query):
    try:
        response = openai.Embedding.create(  # Sử dụng openai.embeddings thay vì openai.Embedding.create
            engine="text-embedding-3-large", 
            input=query
        )
        return np.array(response['data'][0]['embedding'])
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Đọc file JSON chứa dữ liệu kỹ năng và danh mục
@st.cache_data
def load_skill_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Đọc file JSON chứa câu hỏi và embedding
@st.cache_data
def load_question_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for item in data:
        item['embedding'] = np.array(item['embedding'])
    return data

# Giao diện Streamlit
st.title("Tìm kiếm câu hỏi theo danh mục và kỹ năng")
st.write("Chọn danh mục, kỹ năng và nhập truy vấn của bạn để tìm kiếm.")

# Đường dẫn đến file JSON
skill_data_file = './data.json'
question_data_file = './combined_data.json'

# Tải dữ liệu
skill_data = load_skill_data(skill_data_file)
question_data = load_question_data(question_data_file)

# Tạo danh mục lựa chọn
categories = [category['category'] for category in skill_data['value']]
selected_category = st.selectbox("Chọn danh mục:", options=categories)

# Lựa chọn kỹ năng (multiple)
skills = []
for category in skill_data['value']:
    if category['category'] == selected_category:
        skills = category['skills']
        break
selected_skills = st.multiselect("Chọn kỹ năng:", options=skills)

# Tạo câu truy vấn tự động dựa trên lựa chọn
query = f"Search for 5 questions with category '{selected_category}' and skills {', '.join(selected_skills)} about:"

# Hiển thị câu truy vấn vào ô tìm kiếm
query_input = st.text_input("Nhập truy vấn:", value=query)

# Tạo FAISS index từ dữ liệu câu hỏi
def create_faiss_index(data):
    embeddings = np.array([item['embedding'] for item in data])
    dimension = embeddings.shape[1]  # Kích thước của vector
    index = faiss.IndexFlatL2(dimension)  # Tạo FAISS index với khoảng cách L2 (Euclidean)
    index.add(embeddings)  # Thêm embeddings vào index
    return index

# Tìm kiếm câu hỏi tương tự với FAISS
def find_top_questions_faiss(query_embedding, index, data, top_n=5):
    distances, indices = index.search(np.array([query_embedding]), top_n)  # Tìm kiếm các câu hỏi tương tự nhất
    top_questions = [data[idx] for idx in indices[0]]  # Lấy câu hỏi từ chỉ số
    return top_questions

# Lọc câu hỏi theo category và kỹ năng
def filter_questions_by_category_and_skill(questions, category, skills):
    filtered_questions = []
    for question in questions:
        # Kiểm tra điều kiện lọc category và skills
        if category in question.get('category', []) and any(skill in question.get('skills', []) for skill in skills):
            filtered_questions.append(question)
    return filtered_questions

# Tìm kiếm
if st.button("Tìm kiếm"):
    if not query_input.strip():
        st.warning("Vui lòng nhập truy vấn để tìm kiếm.")
    else:
        st.write(f"Đang xử lý truy vấn: **{query_input}**")
        
        # Lọc câu hỏi theo category và skills
        filtered_questions = filter_questions_by_category_and_skill(question_data, selected_category, selected_skills)
        
        if filtered_questions:
            # Tạo FAISS index từ câu hỏi đã lọc
            faiss_index = create_faiss_index(filtered_questions)
            
            query_embedding = get_query_embedding(query_input)
            if query_embedding is not None:
                top_questions = find_top_questions_faiss(query_embedding, faiss_index, filtered_questions)
                
                if top_questions:
                    st.write("### Top 5 câu hỏi tương tự nhất:")
                    for idx, item in enumerate(top_questions, 1):
                        st.write(f"**Câu hỏi {idx}:** {item['question']}")
                        st.write(f"**Danh mục:** {', '.join(item.get('category', []))}")
                        st.write(f"**Kỹ năng liên quan:** {', '.join(item.get('skills', []))}")
                        st.write("**Lựa chọn:**")
                        options = item.get('options', [])
                        for option in options:
                            is_answer_key = "✅" if option.get("isAnswerKey", False) else "❌"
                            st.write(f"- {option['description']} {is_answer_key}")
                        st.write("---")
                else:
                    st.write("Không tìm thấy câu hỏi tương tự.")
            else:
                st.error("Không thể tạo embedding cho truy vấn.")
        else:
            st.write("Không có câu hỏi nào thuộc category và skill bạn chọn.")
