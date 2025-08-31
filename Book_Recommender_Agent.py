import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import os

load_dotenv()

oak = os.getenv("OPENAI_API_KEY")
#system prompt to keep the agent moduler
template = """
You are an AI book recommendation assistant.
Only answer book-related questions. 
If the question is unrelated to books, politely redirect to book topics.

Here is the relevant memory/context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)
#embedd my inputs using openai
embeddings = OpenAIEmbeddings()
#store them in the Chroma vector database
vectorstore = Chroma(collection_name="user_memory", embedding_function=embeddings, persist_directory="./chroma_db")

#manages dynamic memory of the chatbot agent
def add_memory(input_text):
    vectorstore.add_texts([input_text])
def persist():
    vectorstore.persist()

#add a retriever so that the agent references the stored embeddings
retriever = vectorstore.as_retriever(search_type="similarity", search_k=5)
llm = ChatOpenAI(temperature=0.7, openai_api_key=oak)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True

)

# --- Streamlit UI ---
st.title("📚 AI Book Recommender")
st.write("Tell me what you're into, and I'll recommend books!")

# Keep chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_message = st.chat_input("Ask for recommendations or tell me what you read:")

if user_message:
    # 1. Add user input to memory
    add_memory(user_message)
    # 2 Persist embedding locally so memory is retained after ending run
    persist()
    # 2. Get AI response
    answer = qa({"question": user_message, "chat_history": st.session_state.chat_history})
    # 3. Save to session chat history
    st.session_state.chat_history.append(("user", user_message))
    st.session_state.chat_history.append(("ai", answer["answer"]))

# Render chat history as bubbles
for role, msg in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)