from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from colorama import Fore
import os
import warnings 

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()
os.getenv("OPENAI-API-KEY")

# Load the model
llm = ChatOpenAI()

# Load the documents
loader = WebBaseLoader(
    "https://python.langchain.com/docs/get_started/introduction"
)
documents = loader.load()
# print(documents[0])

# prompt templates
template = """You're a senior developer who
answers {question} based on your knowledge and {context}
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"])

# RETRIEVER - Load embeddings and create a vector store 
embeddings = OpenAIEmbeddings() 
db = FAISS.from_documents(documents, embeddings)

# GENERATE - Define the function to generate the response
def generate(query: str):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)
    
    
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=db.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    return result


def start():
    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:

            response = generate(user_input)

            print(Fore.BLUE + f"A: " + response['answer'] + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
