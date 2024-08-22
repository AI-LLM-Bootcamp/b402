import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")

loaded_data = loader.load()

print("\n----------\n")

print("TXT file loaded:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

print("Content of the first page loaded:")

print("\n----------\n")
#print(loaded_data[0].page_content)

print("\n----------\n")

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([loaded_data[0].page_content])

print("\n----------\n")

print("How many chunks of text were created by the splitter?")

print("\n----------\n")
#print(len(texts))

print("\n----------\n")

print("Print the first chunk of text")

print("\n----------\n")
#print(texts[0])

print("\n----------\n")

metadatas = [{"chunk": 0}, {"chunk": 1}]

documents = text_splitter.create_documents(
    [loaded_data[0].page_content, loaded_data[0].page_content], 
    metadatas=metadatas
)

print("\n----------\n")

print("Using a second splitter to create chunks of text with metadata, print the first chunk of text with metadata")

print("\n----------\n")
#print(documents[0])

print("\n----------\n")