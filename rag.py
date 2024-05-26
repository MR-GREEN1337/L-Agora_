from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
import os
from langchain.document_loaders import PyPDFLoader
from langgraph.graph import END, StateGraph
from langchain.schema import Document
from langchain.tools import tool
from utils import load_env

class Advanced_RAG:
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents 
        """
        question : str
        generation : str
        web_search : str
        documents : List[str]
    ### Nodes 

    def retrieve(state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        mistral_model = "mistral-large-latest"
        llm = ChatMistralAI(model=mistral_model, temperature=0)
        pdf_files = [os.path.join("docs/", f) for f in os.listdir("docs/") if f.endswith('.pdf')]
        docs = [PyPDFLoader(file).load() for file in pdf_files]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
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
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
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
        return {"documents": documents, "question": question, "generation": generation}

        
    def web_search(state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = TavilySearchResults(k=3).invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    ### Edges

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        web_search = state["web_search"]
        filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
    @tool
    def kickoff(prompt):
        """Adaptive RAG"""

        load_env()
        # Data model
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
        workflow = StateGraph(Advanced_RAG.GraphState())

        # Define the nodes
        workflow.add_node("retrieve", Advanced_RAG.retrieve)  # retrieve
        workflow.add_node("grade_documents", Advanced_RAG.grade_documents)  # grade documents
        workflow.add_node("generate", Advanced_RAG.generate)  # generatae
        workflow.add_node("websearch", Advanced_RAG.web_search)  # web search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            Advanced_RAG.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", END)

        # Compile
        app = workflow.compile()
        output = ""
        inputs = {"question": prompt}
        for output in app.stream(inputs):
            for key, value in output.items():
                values += value["generation"]
        
        return values