import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

# Original version: this example used gpt-3.5-turbo-0125.
# OpenAI has since retired it. Today we use a current chat model.
MODEL = "gpt-6-luna"  # small, fast and affordable current OpenAI model

# This version of langchain-openai sends temperature=0.7 by default,
# but recent OpenAI models only accept the default value (1).
# That is why we set temperature=1 explicitly.
chatModel = ChatOpenAI(model=MODEL, temperature=1)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")

loaded_data = loader.load()

print("\n----------\n")

print("Loaded TXT file:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('./data/Street_Tree_List.csv')

loaded_data = loader.load()

print("\n----------\n")

print("Loaded CSV file:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import UnstructuredHTMLLoader

# UnstructuredHTMLLoader uses the unstructured library, which needs some
# NLTK language packages. This version of unstructured tries to download
# them from an address that is no longer available (HTTP Error 403).
# We download them from NLTK's official server instead. If they are already
# on your computer, NLTK skips the download.
import nltk

for package in ["punkt", "punkt_tab",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(package, quiet=True)

loader = UnstructuredHTMLLoader('./data/100-startups.html')

loaded_data = loader.load()

print("\n----------\n")

print("Loaded HTML page:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./data/5pages.pdf')

loaded_data = loader.load_and_split()

print("\n----------\n")

print("Loaded HTML page:")

print("\n----------\n")
#print(loaded_data[0].page_content)

print("\n----------\n")

from langchain_community.document_loaders import WikipediaLoader

# WikipediaLoader uses the wikipedia library, which identifies itself to
# Wikipedia with the same generic User-Agent for everyone. Wikipedia now
# limits requests with that User-Agent and answers with a page that is not
# JSON ("JSONDecodeError: Expecting value"). Wikimedia asks each app to send
# its own User-Agent with a way to contact you, so we set one here.
# Replace the email with your own.
import wikipedia

wikipedia.set_user_agent("llm-apps-bootcamp-exercise/1.0 (your-email@example.com)")

name = "JFK"

loader = WikipediaLoader(query=name, load_max_docs=1)

loaded_data = loader.load()[0].page_content

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("human", "Answer this {question}, here is some extra {context}"),
    ]
)

messages = chat_template.format_messages(
    question="What was the full name of JFK?",
    context=loaded_data
)

response = chatModel.invoke(messages)

print("\n----------\n")

print("Respond from Wikipedia: What was the full name of JFK?")

print("\n----------\n")
#print(response.content)

print("\n----------\n")
