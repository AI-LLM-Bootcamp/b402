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

from langchain_openai import OpenAIEmbeddings

# Original version: OpenAIEmbeddings() used its default model,
# text-embedding-ada-002. That model still works, but OpenAI now presents
# text-embedding-3-small as its current small embedding model: it performs
# a bit better and costs less. Its vectors have the same length (1536).
# Writing the model name makes it easy to see and change.
EMBEDDING_MODEL = "text-embedding-3-small"

embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

chunks_of_text =     [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]

embeddings = embeddings_model.embed_documents(chunks_of_text)

print("\n----------\n")

print("How many embeddings were created?")

print("\n----------\n")
#print(len(embeddings))

print("\n----------\n")

print("How long is the first embedding?")

print("\n----------\n")
#print(len(embeddings[0]))

print("\n----------\n")

print("Print the first 5 elements of the first embedding:")

print("\n----------\n")
#print(embeddings[0][:5])

print("\n----------\n")

embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
