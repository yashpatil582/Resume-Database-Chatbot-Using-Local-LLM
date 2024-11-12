import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF
import logging


logging.basicConfig(level=logging.INFO)
# Set up the page configuration for the Streamlit app
st.set_page_config(page_title="Resume Chatbot and Enhancer")

def display_messages():
    """
    Displays chat messages in the Streamlit app.
    """
    # Display a subheader for the chat
    st.subheader("Chat with Your Resume")

    # Iterate through messages stored in the session state
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        # Display each message using the message function with appropriate styling
        message(msg, is_user=is_user, key=str(i))

    # Create an empty container for a thinking spinner
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """
    Processes user input and updates the chat messages in the Streamlit app.
    """
    # Check if there is user input and it is not empty
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        # Extract and clean the user input
        user_text = st.session_state["user_input"].strip()

        # Display a thinking spinner while the assistant processes the input
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            # Ask the assistant for a response based on the user input
            agent_text = st.session_state["assistant"].ask(user_text)

        # Append user and assistant messages to the chat messages in the session state
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

        # Clear the user input field
        st.session_state["user_input"] = ""

def read_and_save_file():
    """
    Reads and saves the uploaded PDF file, performs ingestion, and clears the assistant state.
    """
    # Clear the state of the question-answering assistant
    st.session_state["assistant"].clear()

    # Clear the chat messages and user input in the session state
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Iterate through the uploaded files in the session state
    for file in st.session_state["file_uploader"]:
        # Save the file to a temporary location and get the file path
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Display a spinner while ingesting the file
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

def enhance_resume():
    """
    Enhances the uploaded resume based on the provided job description.
    """
    job_description = st.session_state.get("job_description", "").strip()
    resume_file = st.session_state.get("resume_file")

    if not job_description:
        st.warning("Please enter the job description.")
        return

    if not resume_file:
        st.warning("Please upload your resume.")
        return

    # Save the resume file to a temporary location and get the file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(resume_file.getbuffer())
        resume_file_path = tf.name

    # Extract text from the resume PDF using PyPDFLoader
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path=resume_file_path)
    data = loader.load()

    # Combine the text from all pages
    resume_text = "\n".join([page.page_content for page in data])

    # Display a spinner while enhancing the resume
    with st.spinner("Enhancing your resume..."):
        # Get the recommendations as plain text
        enhanced_resume = st.session_state["assistant"].enhance_resume(
            resume_text, job_description
        )

    # Display the enhanced resume
    st.subheader("Recommendations to Enhance Your Resume")
    st.text_area("Recommendations:", value=enhanced_resume, height=400)

    # Provide a download button for the recommendations
    st.download_button(
        label="Download Recommendations",
        data=enhanced_resume,
        file_name="resume_recommendations.txt",
        mime="text/plain",
    )
 


def page():
    """
    Defines the content of the Streamlit app page for ChatPDF and Resume Enhancement.
    """
    # Check if the session state is empty (first time loading the app)
    if "messages" not in st.session_state:
        # Initialize the session state with empty chat messages and a ChatPDF assistant
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()
        st.session_state["user_input"] = ""

    # Display the main header of the Streamlit app
    st.header("Resume Chatbot and Enhancer")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Chat with Your Resume", "Enhance Your Resume"])

    with tab1:
        # Display a subheader and a file uploader for uploading PDF files
        st.subheader("Upload a PDF Resume to Chat")
        st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        # Create an empty container for a spinner related to file ingestion
        st.session_state["ingestion_spinner"] = st.empty()

        # Display chat messages in the Streamlit app using the defined function
        display_messages()

        # Display a text input field for user messages in the Streamlit app
        st.text_input("Message", key="user_input", on_change=process_input)

    with tab2:
        # Display inputs for job description and resume upload
        st.subheader("Enhance Your Resume Based on Job Description")

        st.text_area(
            "Enter the Job Description:",
            key="job_description",
            height=200,
        )

        st.file_uploader(
            "Upload Your Resume (PDF):",
            type=["pdf"],
            key="resume_file",
            accept_multiple_files=False,
        )

        if st.button("Enhance Resume"):
            enhance_resume()

# Check if the script is being run as the main module
if __name__ == "__main__":
    # Call the "page" function to set up and run the Streamlit app
    page()
