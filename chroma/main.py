import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# https://python.langchain.com/docs/modules/data_connection/vectorstores/

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGUAGE_MODEL = "gpt-3.5-turbo-instruct"

template: str = """/
    You are a customer support specialist /
    question: {question}. You assist users with general inquiries based on {context} /
    and  technical issues. /
    """
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="{question}",
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

model = ChatOpenAI()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def load_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    raw_documents = TextLoader("./docs/user-manual.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    return text_splitter.split_documents(raw_documents)


def load_embeddings(documents, user_query):
    """Create a vector store from a set of documents."""
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    docs = db.similarity_search(user_query)
    return db.as_retriever()


def generate_response(retriever, query):
    pass
    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    # create the prompt

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | model
        | StrOutputParser()
    )
    return chain.invoke(query)


def query(query):
    documents = load_documents()
    retriever = load_embeddings(documents, query)
    response = generate_response(retriever, query)
    return response


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

            response = query(user_input)

            print(Fore.BLUE + f"A: " + response + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
