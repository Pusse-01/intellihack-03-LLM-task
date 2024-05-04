### README.md for Smart Bank Loan Support Chatbot

This README provides the necessary instructions to set up and run the Smart Bank Loan Support Chatbot, a Streamlit-based application designed to assist users with various loan services using a conversational AI powered by large language models.

#### Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- Other dependencies listed in the `requirements.txt` file

#### Installation

1. **Clone the Repository**

   Begin by cloning the repository to your local machine:

   ```bash
   git clone [URL to the repository]
   cd [repository name]
   ```

2. **Set Up a Virtual Environment** (Optional but recommended)

   Create a virtual environment to manage dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install all required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**

   You need to set up the OpenAI API key as an environment variable. Replace `<your_openai_api_key>` with your actual OpenAI API key.

   - On Unix/Linux/macOS:

     ```bash
     export OPENAI_API_KEY=<your_openai_api_key>
     ```

   - On Windows:

     ```bash
     set OPENAI_API_KEY=<your_openai_api_key>
     ```

   Alternatively, you can create a `.env` file in the root directory of the project and add the following line:

   ```
   OPENAI_API_KEY=<your_openai_api_key>
   ```

#### Running the Application

After setting up your environment, you can run the application by executing the following command in the terminal:

```bash
streamlit run app.py
```

This will start the Streamlit server, and the application should be accessible via a web browser at `http://localhost:8501` by default.

#### Usage

The chatbot interface is intuitive:

- **Start a Conversation**: Simply type your question or request related to the loan services in the chat input box.
- **Interact with the AI**: The chatbot will respond with information pulled from the relevant documents or guide you through the loan application process.
- **Clear History**: You can clear the chat history using the sidebar button if needed to start a new session.

#### Additional Notes

- Ensure that the document loader points to the correct location of your loan-related documents.
- The application is configured to use the GPT-3.5 model; ensure your API quota supports this.
- If you encounter any issues with vector storage or retrieval, verify the configuration and compatibility of the FAISS library with your system.

#### Support

For any issues or further assistance, you can contact the development team or raise an issue in the repository.

This should provide a comprehensive guide for setting up and running the Smart Bank Loan Support Chatbot. Ensure you replace placeholders (like `[URL to the repository]`) with actual data relevant to your project.
