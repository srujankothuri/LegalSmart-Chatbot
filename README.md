<h1 align="center">LegalSmart Chatbot</h1>


<p align="center">
<img src="https://github.com/srujankothuri/LegalSmart-Chatbot/blob/e896298cd49582cc06194d088f62dd993409367a/chatbot_legal.jpeg"/>
</p>

## About The Project
LegalSmart Chatbot is a Retrieval-Augmented Generation (RAG)-based generative AI attorney chatbot, designed to assist users by leveraging data from the Indian Penal Code (IPC), the Bharatiya Nyaya Sanhita, and the Constitution of India. Developed using Streamlit, LangChain, and the TogetherAI API for its language model, this tool empowers individuals to understand and exercise their legal rights effectively.

The chatbot provides accurate and contextually relevant responses to legal queries based on the Indian Penal Code (IPC), the Bharatiya Nyaya Sanhita, and the Constitution of India. Whether you are new to understanding your rights, seeking justice under the IPC, or exploring constitutional provisions, LegalSmart Chatbot is tailored to guide you comprehensively across these legal domains.
<br>







## Getting Started

#### 1. Clone the repository:
   - ```
     git https://github.com/srujankothuri/LegalSmart-Chatbot.git
     ```
#### 2. Install necessary packages:
   - ```
     pip install -r requirements.txt
     ```
#### 3. Run the `ingest.py` file, preferably on kaggle or colab for faster embeddings processing and then download the `ipc_vector_db` from the output folder and save it locally.
#### 4. Sign up with Together AI today and get $25 worth of free credit! 🎉 Whether you choose to use it for a short-term project or opt for a long-term commitment, Together AI offers cost-effective solutions compared to the OpenAI API. 🚀 You also have the flexibility to explore other Language Models (LLMs) or APIs if you prefer. For a comprehensive list of options, check out this link: [python.langchain.com/docs/integrations/llms](https://python.langchain.com/docs/integrations/llms) . Once signed up, seamlessly integrate Together AI into your Python environment by setting the API Key as an environment variable. 💻✨ 
   - ```
      os.environ["TOGETHER_API_KEY"] = "YOUR_TOGETHER_API_KEY"`
     ```
   - If you are going to host it in streamlit, huggingface or other...
      - Save it in the secrets variable provided by the hosting with the name `TOGETHER_API_KEY` and key as `YOUR_TOGETHER_API_KEY`.

#### 5. To run the `app.py` file, open the CMD Terminal and and type `streamlit run FULL_FILE_PATH_OF_APP.PY`.

## Contact
If you have any questions or feedback, please raise an [github issue](https://github.com/srujankothuri/LegalSmart-Chatbot/issues)
