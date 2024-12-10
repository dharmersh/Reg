
import bs4
from dotenv import load_dotenv
from langchain import hub
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import format_qa_pair, format_qa_pairs

from colorama import Fore
import warnings
import os
warnings.filterwarnings("ignore")

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI()

# load documents (data source: https://blog.langchain.dev/)
loader = WebBaseLoader("https://blog.langchain.dev/deconstructing-rag/")
docs = loader.load()

loader = WebBaseLoader("https://blog.langchain.dev/reflection-agents/")
docs.extend(loader.load())





def load_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    #  split documents, create vector store and load embeddings
    raw_documents = WebBaseLoader(
        web_paths=("https://blog.langchain.dev/reflection-agents/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("article-header section", "article-header__content", "article-header__footer")
            )
        ),
    )
    blog_docs = raw_documents.load()
    
   # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, 
        chunk_overlap=50)

    return text_splitter.split_documents(blog_docs)


def load_embeddings(documents, user_query):
    """Create a vector store from a set of documents."""
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    #docs = db.similarity_search(user_query)
    return db.as_retriever()
    

# 1. DECOMPOSITION
template = """You are a helpful assistant trained to generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)


def generate_sub_questions(query):
    """ generate sub questions based on user query"""
    pass 
    # Chain
    generate_queries_decomposition = (
        prompt_decomposition 
        | llm 
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    ) 

    # Run
    sub_questions = generate_queries_decomposition.invoke({"question": query})
    questions_str = "\n".join(sub_questions)
    print(f"{Fore.GREEN}Sub-questions: {questions_str}")
      


# 2. ANSWER SUBQUESTIONS RECURSIVELY 
template = """Here is the question you need to answer:

\n --- \n {sub_question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {sub_question}
"""
prompt_qa = ChatPromptTemplate.from_template(template)


def generate_qa_pairs(sub_questions):
    """ ask the LLM to generate a pair of question and answer based on the original user query """
    pass
    

# 3. ANSWER INDIVIDUALLY

# RAG prompt = https://smith.langchain.com/hub/rlm/rag-prompt
prompt_rag = hub.pull("rlm/rag-prompt")


def retrieve_and_rag(prompt_rag, sub_questions):
    """RAG on each sub-question"""
    pass
    


# SUMMARIZE AND ANSWER 

# Prompt
template = """Here is a set of Q+A pairs:

{context}

to use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Query
def query(query):
    documents = load_documents()
    reteriever = load_embeddings(documents,query)
    # generate optimized answer for a given query using the improved subqueries
    question = "What are the main components of an LLM-powered autonomous agent system?"
    queries = [
        "How is context improving AI systems",
        "What are the two main components involved in Basic Reflection",
        "Explain the steps involved in the Reflexion loop"
    ]
    generate_sub_questions(queries[0])
    
query("skipt")