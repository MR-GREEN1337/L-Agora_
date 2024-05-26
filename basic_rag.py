from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.tools import tool

@tool
def simple_rag(question):
    """Simplest RAG"""
    mistral_model = "mistral-large-latest"
    llm = ChatMistralAI(model=mistral_model, temperature=0)
    pdf_files = [os.path.join("docs/", f) for f in os.listdir("docs/") if f.endswith('.pdf')]
    docs = [PyPDFLoader(file).load() for file in pdf_files]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    prompt = hub.pull("rlm/rag-prompt")
    doc_splits = text_splitter.split_documents(docs_list)
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=MistralAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    mistral_model = "mistral-large-latest" # "open-mixtral-8x22b" 
    # Retrieval
    documents = retriever.invoke(question)
    prompt = hub.pull("rlm/rag-prompt")
    mistral_model = "mistral-large-latest"
    # LLM
    llm = ChatMistralAI(model=mistral_model, temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})

    return generation