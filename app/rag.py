import pandas as pd
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import json
import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    persist_directory = "chroma_db"  # Directory for persistent storage

    def __init__(self):
        """
        Initializes the question-answering system with default configurations.
        """
        logging.info("Initializing ChatPDF instance.")
        
        # Initialize the Ollama model with 'llama3.1'.
        self.model_name = "llama3.1"  # Use the exact model name as per your Ollama installation
        self.model = Ollama(model=self.model_name)
        logging.info(f"Ollama model '{self.model_name}' initialized.")

        # Initialize the RecursiveCharacterTextSplitter with specific chunk settings.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        logging.info("Text splitter initialized with chunk size 1000 and overlap 100.")

        # Initialize the PromptTemplate with a predefined template for constructing prompts.
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful HR assistant that analyzes resumes from different candidates.
Use the following pieces of retrieved context to answer the question.
Provide detailed and specific information.

Question: {question}
Context: {context}
Answer:"""
        )
        logging.info("Prompt template initialized.")

        # Prompt template for enhancing the resume
        self.enhance_prompt = PromptTemplate(
            input_variables=["resume", "job_description"],
            template="""
You are an expert career consultant specializing in resume optimization for Applicant Tracking Systems (ATS).

TASK:
- Analyze the candidate's resume.
- Compare it with the job description.
- Identify missing skills, keywords, and experiences.
- Provide specific, actionable recommendations to improve the resume and increase its ATS score.

FORMAT:
Provide the recommendations in a numbered list.

Job Description:
{job_description}

Candidate's Resume:
{resume}

Recommendations:
"""
        )
        logging.info("Enhance resume prompt template initialized.")

    def ingest(self, pdf_file_path: str):
        # Clear the persistent directory if it exists
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            logging.info(f"Cleared existing database directory: {self.persist_directory}")

        # Load the PDF data
        logging.info(f"Loading PDF data from: {pdf_file_path}")
        loader = PyPDFLoader(file_path=pdf_file_path)
        data = loader.load()

        # Combine 'Candidate ID' and 'Resume' into the 'page_content' if necessary
        for doc in data:
            doc.page_content = f"{doc.page_content}"

        # Split the documents into chunks
        logging.info("Splitting documents into chunks.")
        chunks = self.text_splitter.split_documents(data)

        # Create a vector store using embeddings and persistent storage
        logging.info("Creating vector store with embeddings and persistent storage.")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=self.persist_directory  # Use persistent storage
        )

        # Set up the retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        logging.info("Retriever initialized with similarity search.")

        # Save indexed data to JSON for debugging or testing
        indexed_data = [{"id": i, "content": doc.page_content} for i, doc in enumerate(chunks)]
        indexed_file_path = "indexed_data.json"
        with open(indexed_file_path, "w") as f:
            json.dump(indexed_data, f)
        logging.info(f"Indexed data saved to {indexed_file_path}")

        # Define the chain using the prompt and model
        self.chain = self.prompt | self.model
        logging.info("Chain initialized with prompt and model.")

    def ask(self, query: str):
        if not self.chain:
            logging.warning("Chain is not initialized. Please add a PDF document first.")
            return "Please, add a PDF document first."

        # Retrieve relevant context
        logging.info(f"Query received: {query}")
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            logging.warning("No relevant documents found.")
            return "Sorry, I couldn't find any information related to your query."

        # Debug: Print retrieved documents
        for i, doc in enumerate(docs):
            logging.info(f"\n--- Retrieved Document {i+1} ---")
            print(doc.page_content[:500])  # Print first 500 characters

        context = "\n\n".join([doc.page_content for doc in docs])

        # Run the chain with the context and question
        response = self.chain.invoke({'context': context, 'question': query})
        logging.info(f"Response generated: {response}")

        return response

    def enhance_resume(self, resume_text: str, job_description: str):
        """
        Enhances the resume based on the provided job description.
        """
        # Create the prompt input
        prompt_input = self.enhance_prompt.format(
            resume=resume_text,
            job_description=job_description
        )

        # Log the prompt for debugging
        logging.info(f"Enhance Resume Prompt:\n{prompt_input}")

        # Generate the recommendations using the model
        response = self.model.generate([prompt_input])  # Wrap prompt_input in a list

        # Extract the generated text from the response
        recommendations = response.generations[0][0].text  # Access the first generation's text

        logging.info(f"Generated enhancement response:\n{recommendations}")
        return recommendations  # Return the plain text recommendations



    def clear(self):
        """
        Clears the components in the question-answering system.
        """
        logging.info("Clearing vector store, retriever, and chain.")
        
        # Set the vector store to None.
        self.vector_store = None

        # Set the retriever to None.
        self.retriever = None

        # Set the processing chain to None.
        self.chain = None