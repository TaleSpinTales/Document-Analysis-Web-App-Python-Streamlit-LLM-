
# ---------- Imports --------- #
import streamlit as st  # Web framework for building the user interface
from dotenv import load_dotenv  # Loads environment variables from a .env file
from PyPDF2 import PdfReader  # Library for reading PDF files
from langchain.text_splitter import CharacterTextSplitter  # Splits text into manageable chunks
from langchain.embeddings import OpenAIEmbeddings  # Generates vector embeddings using OpenAI
from langchain.vectorstores import FAISS  # In-memory vector database for semantic search
from langchain.chat_models import ChatOpenAI  # OpenAI-powered conversational model (like GPT)
from langchain.memory import ConversationBufferMemory  # Stores the chat history for context
from langchain.chains import ConversationalRetrievalChain  # Combines retrieval and conversation logic
from htmlTemplates import css, bot_template, user_template  # Custom HTML/CSS templates for UI display

import spacy  # Natural Language Processing library for Named Entity Recognition
import requests  # Handles HTTP requests (used for fetching web pages)
from bs4 import BeautifulSoup  # Parses and extracts text from HTML pages

# Load spaCy model globally so it's available everywhere
nlp = spacy.load("en_core_web_sm")

# Logging configuration to capture errors and important events
import logging
# Clear any existing logging configuration (useful in Streamlit which may re-run the script)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Define logging to file
logging.basicConfig(
    handlers=[logging.FileHandler("app_log.txt", mode='a')],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------- PDF ----------- #
def get_pdf_text(pdf_docs):  # Extracts text from a list of uploaded PDF files
    for pdf in pdf_docs:
        if pdf.size > 2 * 1024 * 1024:
            st.error(f"File {pdf.name} exceeds the 2MB size limit.")
            logging.warning(f"Rejected file {pdf.name} due to size > 2MB.")
            return
    text = ""  # Initialize an empty string to store the extracted text
    for pdf in pdf_docs:  # Iterate through each uploaded PDF file
        pdf_reader = PdfReader(pdf)  # Create a PDF reader object for the current file
        for page in pdf_reader.pages:  # Iterate through all pages in the PDF
            if page.extract_text():  # Check if text exists on the page
                text += page.extract_text()  # Append the page's text to the full text
    return text  # Return the combined text from all PDFs

# ----------- URL ----------- #
def get_text_from_url(url):  # Fetches and cleans visible text content from a given webpage URL
    try:
        response = requests.get(url, timeout=10)  # Sends a GET request to the URL with a timeout of 10 seconds
        response.raise_for_status()  # Raises an error if the response status is not 200 (OK)
        soup = BeautifulSoup(response.text, 'html.parser')  # Parses the HTML content using BeautifulSoup
        for tag in soup(["script", "style"]):  # Remove script and style tags (not useful for text analysis)
            tag.decompose()  # Deletes the tag from the soup object
        text = soup.get_text(separator='\n')  # Extracts all visible text, separating lines with newlines
        clean_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())  # Removes empty and whitespace-only lines
        return clean_text  # Returns the cleaned text extracted from the page

    except Exception as e:  # Catch any errors (e.g., invalid URL, network failure)
        st.error(f"❌ Failed to process URL:\n{e}")  # Display error in the Streamlit interface
        return ""  # Return empty string on failure

# ----------- Text Chunking ----------- #
def get_text_chunks(text):  # Splits a long text string into smaller, manageable chunks
    text_splitter = CharacterTextSplitter(  # Initializes a text splitter object with custom settings
        separator="\n",           # Split chunks based on newline characters
        chunk_size=1000,          # Each chunk will have a maximum of 1000 characters
        chunk_overlap=200,        # Overlap 200 characters between chunks for better context retention
        length_function=len       # Use the built-in len() function to measure chunk length
    )
    chunks = text_splitter.split_text(text)  # Split the input text into chunks based on the configuration
    return chunks  # Return the list of text chunks

# ----------- Vectorstore ----------- #
def get_vectorstore(chunks):  # Converts text chunks into vector embeddings and stores them in a FAISS index
    embeddings = OpenAIEmbeddings()  # Initializes the embedding model using OpenAI (e.g., text-embedding-ada-002)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)  # Creates a FAISS vector database from the embedded text chunks
    return vectorstore  # Returns the vectorstore object for semantic search and retrieval

# ----------- Chat Chain ----------- #
def get_conversation_chain(vectorstore):  # Creates a conversational retrieval chain that enables question-answering over document content
    llm = ChatOpenAI()  # Initializes the OpenAI chat model (e.g., GPT-3.5) for generating responses

    memory = ConversationBufferMemory(  # Sets up memory to retain the full chat history
        memory_key='chat_history',      # Stores the history under the key 'chat_history'
        return_messages=True            # Ensures the memory returns messages in chat format (not just text)
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(  # Combines the LLM, retriever, and memory into a single chain
        llm=llm,                                # The language model used for generating responses
        retriever=vectorstore.as_retriever(),   # Converts the FAISS vector store into a retriever for semantic search
        memory=memory                           # Enables context-aware conversation using the memory
    )
    return conversation_chain  # Returns the full conversation chain for use during chat

def process_user_question(user_question, conversation_chain):
    """
    Analyzes user intent and formats a smart prompt for follow-up question, definition, simplification, or quiz.
    """
    question_lower = user_question.lower()
    prompt = ""

    # Log the incoming question
    logging.info(f"User question received: {user_question}")

    if "define" in question_lower or "definition" in question_lower:
        prompt = f"Provide a clear definition of the following term based on the document:\n{user_question}"
    elif "explain like i'm 5" in question_lower or "simplify" in question_lower or "explain to a child" in question_lower:
        prompt = f"Explain this simply as if speaking to a young child:\n{user_question}"
    elif "quiz" in question_lower or "test me" in question_lower:
        prompt = f"Create a short 3-question quiz based on the document, and include correct answers."
    else:
        prompt = f"Answer the following based on the document content:\n{user_question}"

    # Log the constructed prompt
    logging.info(f"Constructed prompt sent to LLM: {prompt}")

    try:
        response = conversation_chain({'question': prompt})
        st.session_state.chat_history = response['chat_history']
        logging.info(f"LLM response received successfully.")
    except Exception as e:
        logging.error(f"Error in conversation_chain: {e}")
        st.error("An error occurred while processing your question.")
        return  # Stop function if there’s an error

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    #Log the last bot response (most recent)
        if len(st.session_state.chat_history) >= 2:
            last_response = st.session_state.chat_history[-1].content
            confidence = "High" if len(last_response) > 300 else "Medium"
            st.markdown(f"**Confidence Level:** {confidence}")
            logging.info(f"Confidence estimated as: {confidence}")
            
# ----------- User Input Handler ----------- #
def handle_userinput(user_question):  # Handles the user's question and displays the chat interaction
    response = st.session_state.conversation({'question': user_question})  # Sends the question to the conversation chain and gets a response
    st.session_state.chat_history = response['chat_history']  # Saves the full chat history in the session state

    for i, message in enumerate(st.session_state.chat_history):  # Loops through the chat history
        if i % 2 == 0:  # Even index = user message
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  # Display user's message using the user template
        else:  # Odd index = bot (AI) message
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  # Display bot's message using the bot template

# ----------- Named Entity Extraction ----------- #
def extract_entities(text):  
    """
    Extracts key named entities such as names, dates, locations, and organizations from the text.
    """
    doc = nlp(text)  # Process the text using spaCy NLP pipeline
    entities = {
        "Names": [],
        "Dates": [],
        "Locations": [],
        "Organizations": []
    }

    for ent in doc.ents:  # Loop over all recognized entities
        if ent.label_ == "PERSON":  # Recognized as a person's name
            entities["Names"].append(ent.text)
        elif ent.label_ == "DATE":  # Recognized as a date
            entities["Dates"].append(ent.text)
        elif ent.label_ == "GPE":  # Recognized as a location (Geo-Political Entity)
            entities["Locations"].append(ent.text)
        elif ent.label_ == "ORG":  # Recognized as an organization
            entities["Organizations"].append(ent.text)

    # Remove duplicates by converting each list to a set, then back to a list
    return {k: list(set(v)) for k, v in entities.items()}

# ----------- Bullet or Numbered List Extraction ----------- #
def extract_lists(text):
    """
    Extracts lines that appear to be part of lists (bullets, numbers, symbols).
    """
    lists = []  # Initialize empty list to store detected list lines
    for line in text.splitlines():  # Loop through each line of the document
        if line.strip().startswith(("-", "*", "•", "·", "▪", "►")) or line.strip()[:2].isdigit():
            lists.append(line.strip())  # Add lines that start with bullets or numbers
    return lists            

def transform_content(raw_text, style, audience, action, section=None, language=None):
    """
    Transforms the document text based on user-selected action:
    rewrite, expand, translate, or create story.
    """
    prompt = ""  # Initialize the prompt string to send to the LLM

    if action == "Rewrite":
        # Create a prompt to rewrite the full text in a specific style and for a specific audience
        prompt = f"Rewrite the following text in a {style} style for a {audience} audience:\n\n{raw_text}"

    elif action == "Expand":
        # Create a prompt to expand a specific section, or the full text if no section is given
        if section:
            prompt = f"Expand and elaborate the following section for {audience} audience:\n\n{section}"
        else:
            prompt = f"Expand and elaborate the following text for {audience} audience:\n\n{raw_text}"

    elif action == "Translate":
        # Create a prompt to translate the full text into the selected language
        prompt = f"Translate the following text to {language}:\n\n{raw_text}"

    elif action == "Create Fictional Story":
        # Create a prompt to write a fictional story based on the facts from the original text
        prompt = f"Write a fictional story based on the facts from this document:\n\n{raw_text}"

    else:
        # Fallback in case no valid action was selected
        prompt = raw_text

    llm = ChatOpenAI(temperature=0.7)  # Initialize the language model with creative temperature
    response = llm.predict(prompt)  # Generate the result using the language model
    return response  # Return the model's response

def compare_documents(docs_texts):
    """
    Compares multiple documents by analyzing similarities, differences, key arguments, and themes.
    Returns a summarized comparison report.
    """
    llm = ChatOpenAI(temperature=0.5)  # Use moderate temperature for balanced analysis

    # Start building the comparison prompt
    combined_prompt = "You are a document comparison assistant.\n"
    combined_prompt += "Compare the following documents and summarize key similarities, differences, arguments, and data:\n\n"

    # Add each document to the prompt
    for i, text in enumerate(docs_texts):
        combined_prompt += f"Document {i + 1}:\n{text[:2000]}\n\n"  # Limit each to first 2000 chars

    # Add instructions for output structure
    combined_prompt += (
        "Provide a structured comparison highlighting:\n"
        "- Shared topics and overlapping ideas\n"
        "- Major differences in tone, argument, or conclusions\n"
        "- Key data points mentioned in each\n"
        "- Summary insights or synthesis across documents\n"
    )

    # Call the LLM and return the response
    response = llm.predict(combined_prompt)
    return response

def analyze_document_topics(text):
    """
    Analyzes the document to identify main topics, themes, and writing style.
    Returns a summary of what the document is about.
    """
    llm = ChatOpenAI(temperature=0.3)  # Lower temperature for analytical clarity
    prompt = (
        "Analyze the following document and extract its key topics, themes, "
        "and writing style. Provide a concise summary:\n\n" + text[:3000]
    )
    return llm.predict(prompt)  # Return the summary of the document


def generate_recommendation(user_profile, doc_summary):
    """
    Generates a personalized recommendation based on user profile and document summary.
    Includes justification for the recommendation.
    """
    llm = ChatOpenAI(temperature=0.5)  # Moderate creativity for explanation
    prompt = (
        f"The user has the following profile:\n{user_profile}\n\n"
        f"The document summary is:\n{doc_summary}\n\n"
        "Based on the user's profile and the document content, provide a personalized "
        "recommendation whether this document is relevant to the user and explain why."
    )
    return llm.predict(prompt)

def generate_code_documentation(code_text):
    """
    Generates technical documentation for a code snippet or file.
    Includes purpose, functions, parameters, usage examples in Markdown format.
    """
    llm = ChatOpenAI(temperature=0)  # Zero creativity for strict factual output

    prompt = (
        "Analyze the following code and generate clear, concise documentation in Markdown format. "
        "Explain its purpose, key functions, classes, parameters, and usage examples:\n\n"
        + code_text[:3000]  # Limit to first 3000 characters for performance
    )

    documentation = llm.predict(prompt)
    return documentation

def translate_and_localize(text, target_languages):
    """
    Translates and localizes the given text into multiple target languages.
    Highlights any cultural or linguistic adaptations made during localization.
    """
    llm = ChatOpenAI(temperature=0.3)  # Slight creativity for idioms/slang handling
    results = {}

    for lang in target_languages:
        # Prompt for cultural localization
        prompt = (
            f"Translate and localize the following text to {lang}. "
            "Make sure the translation is culturally appropriate, adapting idioms, slang, and references. "
            "Also, highlight any localization changes you made compared to the original text.\n\n"
            f"Original text:\n{text}"
        )

        # Get localized translation
        localized_text = llm.predict(prompt)
        results[lang] = localized_text

    return results  # Dictionary with language as key and localized output as value

# ----------- Main App ----------- #
def main():  # Main function that initializes the app and handles UI logic
    nlp = spacy.load("en_core_web_sm")  # Load English NLP model once for entity extraction
    load_dotenv()  # Loads environment variables from the .env file (e.g., the OpenAI API key)
    st.set_page_config(page_title="Chat with documents and URLs", page_icon=":books:")  
    # Sets the title and icon of the Streamlit app tab
    st.write(css, unsafe_allow_html=True)  
    # Injects custom CSS styles for layout and chat message formatting
    if "conversation" not in st.session_state:  
        st.session_state.conversation = None  
    # Initializes the conversation object in session state if not already set
    if "chat_history" not in st.session_state:  
        st.session_state.chat_history = None  
    # Initializes the chat history in session state if not already set
    st.markdown("<h1 style='text-align: center;'>📖 Automated Document Actions and Question Answering</h1>", unsafe_allow_html=True)  
    # Displays a centered header title using markdown and HTML
    user_question = st.text_input("Ask about the file or link:")  
    # Creates a text input field for the user to ask a question about the uploaded file or link
    if user_question:  # Checks if the user has entered a question
        if st.session_state.conversation:  # Ensures the conversation chain is initialized
            process_user_question(user_question, st.session_state.conversation)
        else:
            # Prevent user from asking questions before uploading a file or URL
            st.warning("Please upload a file or enter a URL and click 'Summarize and Start Chat' before asking questions.")  
            # Shows a warning if the user tries to ask before content is processed

    with st.sidebar:  # Defines the content of the sidebar in the Streamlit app
        st.subheader("📂 Upload PDF")  # Subheader for PDF file upload section
        pdf_docs = st.file_uploader(  
            "Choose File", accept_multiple_files=True  # Allows uploading one or more PDF files
        )
        st.subheader("🌐 Enter URL")  # Subheader for URL input section
        web_url = st.text_input("Type a URL")  # Input field for user to enter a webpage URL
        if st.button("Submit"):  # Button to trigger processing of the document or URL
            with st.spinner("📄 Loading..."):  # Show a spinner while processing
                raw_text = ""  # Initialize an empty string to store the extracted text

                if pdf_docs:  # If PDF files were uploaded
                    raw_text = get_pdf_text(pdf_docs)  # Extract text from the PDF files
                elif web_url:  # If a URL was provided
                    raw_text = get_text_from_url(web_url)  # Extract text from the web page
                else:
                    st.warning("⚠️ Please upload a file or enter a URL")  # Show warning if nothing was provided
                    return  # Exit early

                if raw_text:  # If text was successfully extracted
                    with st.expander("🔍 Insight Extractor (Entities & Lists)"):  # Expandable section for entity and list output
                        entities = extract_entities(raw_text)  # Call function to extract named entities
                        lists = extract_lists(raw_text)  # Call function to extract potential bullet/numbered lists

                        st.subheader("Named Entities:")  # Section title for entity output
                        for category, items in entities.items():  # Display each entity type
                            st.markdown(f"**{category}:** {', '.join(items) if items else 'None'}")  # Print values or 'None'

                            st.subheader("Detected Lists:")  # Section title for list output
                    if lists:
                        for item in lists:
                            st.markdown(f"- {item}")  # Render each list item as a markdown bullet
                        else:
                            st.markdown("_No lists found in this document._")  # Fallback message if no lists found
                    # Summarize the text
                    llm = ChatOpenAI(temperature=0.5)  # Initialize the OpenAI chat model with moderate creativity
                    summary_text = llm.predict(f"Summarize this text:\n{raw_text[:3000]}")  # Request a summary (limited to 3000 characters)
                    # Show summary in a styled container
                    st.markdown(
                        f"""
                        <div style='
                            background-color: #1e1e1e;
                            color: white;
                            padding: 15px;
                            border-radius: 10px;
                            margin-top: 10px;
                            font-size: 16px;
                            text-align: left;
                            line-height: 1.6;
                        '>
                            <b style='display:block; text-align:center; margin-bottom:10px;'>Summary:</b>
                            {summary_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Start the conversation setup
                    chunks = get_text_chunks(raw_text)  # Split the text into overlapping chunks
                    vectorstore = get_vectorstore(chunks)  # Create vector embeddings and index them using FAISS
                    st.session_state.conversation = get_conversation_chain(vectorstore)  # Initialize the conversational chain with memory

                    with st.expander("🪄 Content Transformer (Rewrite / Expand / Translate / Story)"):
                        st.markdown("Use this tool to creatively rewrite or transform the uploaded content.")

                        # User selects the type of transformation to apply
                        action = st.selectbox("Choose action:", ["Rewrite", "Expand", "Translate", "Create Fictional Story"])

                        # Initialize optional variables
                        style = None
                        audience = None
                        language = None
                        section = None

                        # If user selects rewrite or expand, allow them to choose style and audience
                        if action in ["Rewrite", "Expand"]:
                            style = st.selectbox("Select writing style:", ["formal", "informal", "poetic", "journalistic"])
                            audience = st.selectbox("Select target audience:", ["children", "teenagers", "adults", "experts"])

                        # If user selects translation, request the target language
                        if action == "Translate":
                            language = st.text_input("Enter target language (e.g., Spanish, Hebrew, French)")

                        # If user selects expand, allow them to optionally provide a specific section
                        if action == "Expand":
                            section = st.text_area("Enter the section to expand (optional):")

                        # When user clicks the button, process the request
                        if st.button("Transform Content"):
                            # Validate input for translation
                            if action == "Translate" and not language:
                                st.warning("Please enter a target language.")
                            else:
                                # Perform the transformation and show result
                                with st.spinner("🔄 Transforming content..."):
                                    result = transform_content(
                                        raw_text=raw_text,    # Original full document text
                                        style=style,          # User-selected writing style
                                        audience=audience,    # Target audience for tone/complexity
                                        action=action,        # Type of action selected
                                        section=section,      # Optional section (if expanding)
                                        language=language     # Target language (if translating)
                                    )
                                    st.subheader("🔍 Result:")  # Header for the result section
                                    st.markdown(result)        # Display the result using Markdown
                else:
                    st.error("❌ No content found to process")  # Show error if no text was found

                with st.expander("📑 Document Comparator (Compare Multiple Documents)"):
                    st.markdown("Upload at least two documents or enter multiple URLs to compare them.")

                    # Upload multiple PDFs or enter multiple URLs (one per line)
                    uploaded_files = st.file_uploader("Upload 2 or more PDF files:", accept_multiple_files=True)
                    urls_input = st.text_area("Or enter URLs (one per line):")

                    if st.button("Compare Documents"):
                        docs_texts = []  # Will store extracted text from all sources

                        # Process uploaded PDF files
                        if uploaded_files and len(uploaded_files) >= 2:
                            for pdf_file in uploaded_files:
                                docs_texts.append(get_pdf_text([pdf_file]))

                        # Process URLs (optional)
                        if urls_input:
                            urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
                            for url in urls:
                                text = get_text_from_url(url)
                                if text:
                                    docs_texts.append(text)

                        # Validate that at least 2 documents were collected
                        if len(docs_texts) < 2:
                            st.error("Please provide at least two documents (PDFs or URLs) for comparison.")
                        else:
                            with st.spinner("🔍 Analyzing and comparing documents..."):
                                comparison_result = compare_documents(docs_texts)  # Generate comparison report
                                st.subheader("📝 Comparison Report:")
                                st.markdown(comparison_result)

                                with st.expander("🎯 Personalized Content Recommender"):
                                    st.subheader("Enter your profile")
                                    interests = st.text_area("What are your interests, preferences, and background?", height=100)

                                    st.subheader("Upload or enter document content")
                                    recommender_pdf = st.file_uploader("Upload PDF(s) to analyze", accept_multiple_files=True, key="recommender_pdf")
                                    recommender_url = st.text_input("Or enter a URL", key="recommender_url")

                                    if st.button("Generate Recommendation"):
                                        raw_text = ""

                                        # Load document content from PDF or URL
                                        if recommender_pdf:
                                            raw_text = get_pdf_text(recommender_pdf)
                                        elif recommender_url:
                                            raw_text = get_text_from_url(recommender_url)
                                        else:
                                            st.warning("Please upload a PDF or enter a URL.")
                                            return

                                        # Validate profile
                                        if not interests.strip():
                                            st.warning("Please enter your profile information.")
                                            return

                                        # Perform analysis and recommendation
                                        with st.spinner("📊 Analyzing document and generating recommendation..."):
                                            doc_summary = analyze_document_topics(raw_text)  # Analyze document topics
                                            st.subheader("📝 Document Summary")
                                            st.markdown(doc_summary)

                                            recommendation = generate_recommendation(interests, doc_summary)  # Match against profile
                                            st.subheader("💡 Personalized Recommendation")
                                            st.markdown(recommendation)

                                        # Optional feedback input
                                        st.subheader("🗣️ Feedback")
                                        feedback = st.text_area("Let us know what you think:", height=80)
                                        if st.button("Submit Feedback"):
                                            st.success("Thank you for your feedback!")

                            with st.expander("💻 Code Documentation Generator"):
                                st.subheader("Paste your code or upload a code file")

                                # Input area for code as text
                                code_input = st.text_area("Paste code here:", height=200)

                                # File upload option for code files
                                uploaded_code_files = st.file_uploader("Or upload code file(s)", type=["py", "js", "java", "cpp"], accept_multiple_files=True, key="code_docs")

                                if st.button("Generate Documentation"):
                                    code_text = ""

                                    # If files were uploaded, combine their contents
                                    if uploaded_code_files:
                                        for file in uploaded_code_files:
                                            code_text += file.getvalue().decode("utf-8") + "\n\n"

                                    # If pasted code was entered
                                    elif code_input.strip():
                                        code_text = code_input

                                    else:
                                        st.warning("Please paste code or upload file(s).")
                                        return

                                    # Generate and display documentation
                                    with st.spinner("🛠️ Generating documentation..."):
                                        documentation = generate_code_documentation(code_text)
                                        st.subheader("📄 Generated Documentation")
                                        st.markdown(documentation)


                        with st.expander("🌍 Multilingual Content Creator"):
                            st.subheader("Enter text to translate and localize")

                            # Text input box
                            input_text = st.text_area("Enter the original text:", height=200)

                            # Language selection (multi-choice)
                            target_languages = st.multiselect(
                                "Select target languages:",
                                options=["French", "Spanish", "German", "Chinese", "Japanese", "Arabic", "Russian", "Hebrew"]
                            )

                            # Run translation on click
                            if st.button("Translate and Localize"):
                                if not input_text.strip():
                                    st.warning("Please enter some text.")
                                elif not target_languages:
                                    st.warning("Please select at least one target language.")
                                else:
                                    with st.spinner("🌐 Translating and localizing..."):
                                        results = translate_and_localize(input_text, target_languages)  # Perform translations

                                        # Display results per language
                                        for lang, translation in results.items():
                                            st.subheader(f"📘 {lang} Translation & Localization")
                                            st.markdown(translation)

# ----------- Run App ----------- #
if __name__ == '__main__':
    main()
