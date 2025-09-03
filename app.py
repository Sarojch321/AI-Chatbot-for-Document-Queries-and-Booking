import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import asyncio
import nest_asyncio
import parsedatetime
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering.chain import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

import phonenumbers
import dateparser
from email_validator import validate_email, EmailNotValidError


def validate_phone_number(phone: str) -> bool:
    try:
        parsed = phonenumbers.parse(phone, None)
        return phonenumbers.is_valid_number(parsed)
    except:
        return False

def validate_email_address(email: str) -> bool:
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def extract_date_from_text(text: str) -> str:
    cal = parsedatetime.Calendar()
    time_struct, parse_status = cal.parse(text)

    if parse_status == 1:
        dt = datetime(*time_struct[:6])
        return dt.strftime("%Y-%m-%d")
    else:
        return "Invalid or ambiguous date"


tools = [
    Tool.from_function(
        func=extract_date_from_text,
        name="ParseDate",
        description="Convert text like 'next Monday' or 'tomorrow' to date format"
    ),
    Tool.from_function(
        func=validate_phone_number,
        name="ValidatePhoneNumber",
        description="Validate a phone number string"
    ),
    Tool.from_function(
        func=validate_email_address,
        name="ValidateEmailAddress",
        description="Validate an email address"
    )
]


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vectors_store(text_chunks, api_key):
    # print(f"Number of text chunks: {len(text_chunks)}")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from provided context.
    If answer is not available, respond: "Answer is not available in the context".

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def display_chat(user_question, response_text):
    st.markdown(
        f"""
        <div class="chat-message user">
            <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
            <div class="message">{user_question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
            <div class="message">{response_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def download_history(history):
    df = pd.DataFrame(history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
    st.markdown(href, unsafe_allow_html=True)


def initialize_form_agent(llm):
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

def detect_appointment_trigger(user_input: str) -> bool:
    triggers = ["call me", "book", "appointment", "schedule", "meet"]
    return any(trigger in user_input.lower() for trigger in triggers)

def run_conversational_form(llm):
    agent = initialize_form_agent(llm)

    name = st.text_input("What's your name?")
    if not name:
        return

    email = st.text_input("What's your email?")
    if email and not validate_email_address(email):
        st.warning("Invalid email format.")
        return

    phone = st.text_input("What's your phone number?")
    if phone and not validate_phone_number(phone):
        st.warning("Invalid phone number.")
        return

    date_str = st.text_input("Preferred appointment date? (e.g. next Monday)")
    final_date = extract_date_from_text(date_str)
    if final_date == "Invalid or ambiguous date":
        st.warning("Couldn't understand the date. Try 'next Friday', 'tomorrow', etc.")
        return

    st.success(f"Appointment booked for {name} on {final_date}.\n\nConfirmation sent to {email}.")


def handle_user_input(user_input_text, api_key, pdf_docs, conversation_history):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)

    if detect_appointment_trigger(user_input_text):
        st.info("Booking intent detected. Starting conversational form...")
        run_conversational_form(llm)
        return
    
    pdf_text = get_pdf_text(pdf_docs)
    text_chunks = get_chunks(pdf_text)
    vector_store = get_vectors_store(text_chunks, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input_text)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_input_text}, return_only_outputs=True)

    user_question_output = user_input_text
    response_output = response["output_text"]
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []

    conversation_history.append((user_question_output, response_output, "Google AI", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ",".join(pdf_names)))
    display_chat(user_question_output, response_output)


def main():
    st.set_page_config(page_title="Chat with PDFs + Book Appointments", layout="wide")
    st.title("Chat with PDFs + Book Appointments")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    api_key = st.text_input("Enter your Google API Key", type="password")

    if api_key is None:
        return


    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_chunks(raw_text)
                get_vectors_store(chunks, api_key)
                st.success("PDFs processed and ready!")
        else:
            st.warning("Please upload at least one PDF.")

    user_question = st.text_input("Ask a question or request a call:(call me)")

    if user_question:
        handle_user_input(user_question, api_key, pdf_docs, st.session_state.conversation_history)

    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("Conversation History")
        for question, answer, model, timestamp, pdf_name in reversed(st.session_state.conversation_history):
            display_chat(question, answer)
        download_history(st.session_state.conversation_history)

    st.markdown(
        """
        <style>
            .chat-message {
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
                background-color: #2b313e;
            }
            .chat-message.bot {
                background-color: #475063;
            }
            .chat-message .avatar {
                width: 10%;
            }
            .chat-message .avatar img {
                max-width: 48px;
                border-radius: 50%;
            }
            .chat-message .message {
                width: 90%;
                padding: 0 1.5rem;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
