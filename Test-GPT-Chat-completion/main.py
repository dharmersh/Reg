import dotenv
from openai import OpenAI
from colorama import Fore
import warnings   
import os

warnings.filterwarnings("ignore")
# LOAD ENV VARIABLES
dotenv.load_dotenv()
os.getenv("OPENAI-API-KEY")
model=os.getenv("model")
print(model)

# Load the model
client = OpenAI()

# Define a request
print(Fore.GREEN + "Requesting the model to generate a response..." + Fore.RESET + "\n")

completion = client.chat.completions.create(
  model=model,
  messages=[
    {"role": "system", "content": "You are a funny assistant, skilled in telling 2-sentence jokes about the topic given"},
    {"role": "user", "content":"Tell me a joke about the topic 'Python'"}
  ]
)

print(Fore.BLUE + completion.choices[0].message.content)

