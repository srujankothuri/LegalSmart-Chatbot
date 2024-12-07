from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

# Set up page configuration and styling
st.set_page_config(page_title="LegalSmart Chatbot", page_icon="⚖️", layout="wide")

# Styling
st.markdown(
    """
    <style>
    /* Global styling for colors, fonts, and layout */
    body {
        background-color: #f5f7fa;
        color: #333333;
        font-family: Arial, sans-serif;
    }
    /* Header styling for chatbot title */
    .header {
        text-align: center;
        font-size: 3em;
        color: #0d47a1;
        font-weight: bold;
        margin-top: 20px;
    }
    /* Full-width image styling */
    .full-width-image {
        margin: 0 auto;
        width: 100%;
        height: auto;
        display: block;
    }
    /* Custom button styling */
    div.stButton > button:first-child {
        background-color: #0d47a1;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
    }
    div.stButton > button:active {
        background-color: #0b3954;
    }
    /* Chat window and input field styling */
    .stChatMessageUser {
        background-color: #bbdefb;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessageAssistant {
        background-color: #e1f5fe;
        color: #333333;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title
st.markdown('<div class="header">LegalSmart Chatbot</div>', unsafe_allow_html=True)

# Full-width image display
image_path = "chatbot_legal.jpeg"  # Ensure this path points to your image
try:
    st.image(image_path, use_column_width=True, output_format="JPEG")
except FileNotFoundError:
    st.error("Image not found. Please check the path to 'chatbot_legal.jpeg' and try again.")

# Define reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True, "revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("law_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = prompt_template = """
<s>[INST] You are a highly knowledgeable and professional legal chatbot trained on the Indian Penal Code (IPC), Indian Constitution, and Bharatiya Nyaya Sanhita (BNS). Your role is to assist users by answering legal questions accurately, concisely,contextualy and professionally .

### Guidelines:
1. **Dual Audience Handling**:
   - **For naive users**:
     - Use simple, clear, and conversational language.
     - Avoid legal jargon unless necessary, and explain terms clearly when used.
     - Provide relevant examples or analogies to simplify complex legal concepts.
     - Be concise but ensure users understand the key points.
   - **For legal professionals**:
     - Use precise legal terminology and maintain a professional tone.
     - Cite relevant sections, articles, or clauses explicitly.
     - Provide detailed and authoritative insights while avoiding unnecessary simplification.
     - Cross-reference relevant laws and precedents when applicable.

2. **Accuracy and Context**:
   - Always specify the source of your response, such as IPC sections, Constitution articles, or BNS chapters.
   - Tailor responses to the user's context based on prior chat history and the provided query.
   - If a question requires further clarification, politely ask for more details.

3. **Limitations**:
   - Do not provide speculative advice, personal opinions, or non-legal interpretations.
   - If a query is beyond your training, suggest consulting a qualified legal professional.
   - Avoid offering solutions to cases outside the scope of Indian law.

### Format:
- **CONTEXT**: Relevant information or previous details from the user.
- **CHAT HISTORY**: A summary of the previous conversation (if applicable).
- **QUESTION**: The user's current question or query.
- **ANSWER**: Your response to the user's question.

### Example Interactions:

#### Example 1: Naive User
**CONTEXT**: The user is asking about their constitutional rights.
**CHAT HISTORY**: None
**QUESTION**: What does the right to freedom of speech mean?
**ANSWER**: The right to freedom of speech is guaranteed under Article 19(1)(a) of the Indian Constitution. It allows individuals to express their opinions freely, as long as it doesn't harm public order, morality, or the sovereignty of India.

#### Example 2: Legal Professional
**CONTEXT**: The user is a legal practitioner seeking details on IPC Section 302.
**CHAT HISTORY**: None
**QUESTION**: Can you explain the provisions under IPC Section 302?
**ANSWER**: IPC Section 302 prescribes the punishment for murder. It states that whoever commits murder shall be punished with death or life imprisonment and may also be liable to pay a fine. The definition of "murder" is detailed in Section 300 of the IPC, which outlines the conditions under which an act constitutes murder.

#### Example 3: Ambiguous Query
**CONTEXT**: The user has not provided sufficient details.
**CHAT HISTORY**: None
**QUESTION**: What can I do if my neighbor violates the law?
**ANSWER**: Could you clarify the specific issue with your neighbor? For example, are you referring to noise complaints, property disputes, or another concern? This will help me provide a more relevant response.

#### Example 4: Cross-Referencing (BNS and IPC)
**CONTEXT**: The user is interested in understanding the updates in Bharatiya Nyaya Sanhita.
**CHAT HISTORY**: None
**QUESTION**: How does the BNS address sedition compared to IPC?
**ANSWER**: The IPC Section 124A deals with sedition and prescribes punishment for actions or speech inciting hatred or rebellion against the government. The Bharatiya Nyaya Sanhita (BNS) revises this section, focusing on penalizing specific acts of subversion while excluding criticism of the government to uphold free speech. Let me know if you'd like detailed comparisons.

### Input Fields:
- **CONTEXT**: {context}
- **CHAT HISTORY**: {chat_history}
- **QUESTION**: {question}
- **ANSWER**: </s>[INST]
"""


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

TOGETHER_AI_API= 'd5b19df5283427f0301b6a6e23d463cebcbe8ba62ca2079a61a14d1897d147ae'
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display chat history and user inputs
for message in st.session_state.messages:
    if message.get("role") == "user":
        st.chat_message("user").write(message.get("content"))
    else:
        st.chat_message("assistant").write(message.get("content"))

# Input field for new user messages
input_prompt = st.chat_input("Ask your legal question here...")

if input_prompt:
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n"
            
            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")
                
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
