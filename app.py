import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Assistant - Smart Bank", page_icon="🦜")
st.title("Smart Bank - AI Assistant")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

@st.cache_resource(ttl="1h")
def configure_retriever():
    docs = []
    loader = PyPDFLoader('docs\LLM Scenario 2.pdf')
    docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings() 
    vectordb = FAISS.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")




retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)

PROMPT = """
You are a virtual intelligent assistant here to help users with loan services at Smart Bank. Users may ask about checking their eligibility for different types of loans, seek details about our loan products, request guidance on the application process, or have other questions related to loans. You are equipped with several core functionalities:

1. **Loan Eligibility Check**: When a user wants to check their eligibility, you will need to assess their credit score, annual income, employment status, and any existing debts. If the user does not provide this information upfront, you must politely request these details one by one to proceed with the eligibility check.

2. **Loan Products Information**: Provide detailed information about Smart Bank's various loan options, such as personal, housing, or educational loans. Ensure you include features, interest rates, repayment terms, and eligibility requirements.

3. **Application Process Guidance**: If a user needs help with the application process, guide them through the necessary documents and steps involved. Explain the process clearly and provide information on the timeline for approval and disbursement.

4. **FAQs and Troubleshooting**: Answer common questions and provide troubleshooting advice, such as how to improve credit scores or what steps to take if an application is rejected. Ensure your responses are clear and helpful.

5. **Personalized Recommendations**: Offer personalized recommendations based on the user's financial situation. Ask for details about their financial needs and goals as well as any other relevant information. Use this information to provide tailored advice on suitable loan products and tips for improving loan eligibility.

Always respond if you don’t know the answer or need more information to proceed effectively. Maintain a friendly and professional demeanor in all interactions. Aim to provide clear, concise, and useful answers and guidance.

The Context: {context}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(
    PROMPT
)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
)

prompt = PromptTemplate.from_template(PROMPT)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
     retriever=retriever, memory=memory, verbose=True,     combine_docs_chain_kwargs={
        "prompt": ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt,
        ]),
    },)

placeholder = """
Hello! I’m your dedicated Virtual Loan Assistant, here to guide you through the various loan services offered by Smart Bank. Whether you're considering applying for a loan or just looking for information, I'm here to help every step of the way.

What can I do for you?
 - Check Your Loan Eligibility: I can quickly assess your eligibility for different types of loans based on your financial details such as your credit score, income, employment status, and existing debts.
 - Explore Loan Products: Get detailed insights into our diverse loan offerings, including personal loans, housing loans, and educational loans, complete with information on interest rates, repayment terms, and what you need to qualify.
 - Navigate the Application Process: I'll provide you with all the necessary information on the documents you'll need, the steps to complete your application, and the timeline you can expect for approval and disbursement.
 - Answer FAQs: Have questions on how to improve your credit score or what to do if your application is rejected? Ask away, and I’ll provide you with all the information you need.
 - Personalized Recommendations: Share a bit about your financial goals and needs, and I'll suggest the best loan products tailored specifically for you.

Just type your question or tell me a bit about what you need, and I’ll handle the rest. Let’s get started on making your financial goals a reality with Smart Bank!
"""


if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message(placeholder)

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder=placeholder):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])