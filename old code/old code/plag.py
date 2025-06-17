import streamlit as st
import os
from dotenv import load_dotenv

# LangChain / OpenAI
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangOpenAI

# Парсеры
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# API ключ
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Кастомный стиль =====
st.markdown("""
    <style>
        body {
            background-color: #f9f9fb;
        }
        .main-title {
            font-family: 'Segoe UI', sans-serif;
            font-size: 36px;
            color: #333333;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .desc-text {
            font-size: 16px;
            color: #555;
            margin-bottom: 30px;
        }
        .upload-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 25px;
        }
        .btn-custom {
            background-color: #0057D9;
            color: white;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .btn-custom:hover {
            background-color: #0041a8;
        }
        .result-box {
            background-color: #eef2ff;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            font-size: 16px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Табы: ИИ или Плагиат =====
tab1, tab2 = st.tabs(["🔍 Проверка на использование ИИ", "📚 Проверка на плагиат"])

# ===== Вспомогательная функция =====
def extract_text(file) -> str:
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "".join(page.extract_text() or "" for page in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = DocxDocument(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

# ===== Tab 1: Проверка на ИИ =====
with tab1:
    st.markdown('<div class="main-title">🤖 Проверка на использование ИИ</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-text">Загрузите текстовый документ, чтобы определить, был ли он сгенерирован искусственным интеллектом.</div>', unsafe_allow_html=True)

    with st.container():
        ai_file = st.file_uploader("📄 Загрузите документ (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="ai_file")

    if st.button("📡 Анализировать", key="ai_check"):
        if ai_file:
            with st.spinner("Определение признаков генерации..."):
                text = extract_text(ai_file)
                prompt = f"Определи, был ли следующий текст сгенерирован ИИ. Объясни свой вывод.\n\n{text}"
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.choices[0].message.content
            st.markdown('<div class="result-box">' + result + '</div>', unsafe_allow_html=True)
        else:
            st.warning("Пожалуйста, загрузите документ.")

# ===== Tab 2: Проверка на плагиат =====
with tab2:
    st.markdown('<div class="main-title">📚 Система выявления плагиата</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-text">Загрузите эталонные документы и файл для проверки. Наша система сравнит тексты и выдаст результаты совпадений.</div>', unsafe_allow_html=True)

    with st.container():
        ref_files = st.file_uploader("📁 Эталонные документы (.txt, .pdf, .docx)", accept_multiple_files=True, type=["txt", "pdf", "docx"], key="ref_files")
        target_file = st.file_uploader("📄 Документ для проверки", type=["txt", "pdf", "docx"], key="plag_file")

    if st.button("🔎 Проверить на плагиат", key="plag_check"):
        if ref_files and target_file:
            with st.spinner("Анализируем совпадения..."):
                ref_texts = [extract_text(f) for f in ref_files]
                target_text = extract_text(target_file)

                matches = []
                for i, ref in enumerate(ref_texts):
                    common_words = set(target_text.split()) & set(ref.split())
                    similarity = len(common_words) / max(len(set(target_text.split())), 1)
                    if similarity > 0.05:
                        matches.append(f"⚠️ Совпадения с эталоном #{i+1}: {round(similarity*100, 2)}%")

                if not matches:
                    matches.append("✅ Совпадений не обнаружено.")

            st.markdown('<div class="result-box">' + "<br>".join(matches) + '</div>', unsafe_allow_html=True)
        else:
            st.warning("Загрузите оба типа документов для начала проверки.")
