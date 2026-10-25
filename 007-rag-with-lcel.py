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
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Original version: OpenAIEmbeddings() used its default model,
# text-embedding-ada-002. That model still works, but OpenAI now presents
# text-embedding-3-small as its current small embedding model: it performs
# a bit better and costs less. Its vectors have the same length (1536).
# Writing the model name makes it easy to see and change.
# We use this constant in both vector stores below (Chroma and FAISS).
EMBEDDING_MODEL = "text-embedding-3-small"

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader('./data/state_of_the_union.txt').load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings(model=EMBEDDING_MODEL))

question = "What did the president say about the John Lewis Voting Rights Act?"

response = vector_db.similarity_search(question)

print("\n----------\n")

print("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?")

print("\n----------\n")
#print(response[0].page_content)

print("\n----------\n")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/state_of_the_union.txt")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loaded_document = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_db = FAISS.from_documents(chunks_of_text, embeddings)

retriever = vector_db.as_retriever()

response = retriever.invoke("what did he say about ketanji brown jackson?")

print("\n----------\n")

print("Ask the RAG App with Retriever: What did he say about ketanji brown jackson?")

print("\n----------\n")
#print(response[0].page_content)

print("\n----------\n")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Original version: ChatOpenAI() used the library's default model,
# gpt-3.5-turbo, which OpenAI has since retired. We use the same current
# model defined at the top (MODEL), with temperature=1 for the same reason.
model = ChatOpenAI(model=MODEL, temperature=1)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke("what did he say about ketanji brown jackson?")

print("\n----------\n")

print("Ask the RAG App with LCEL: What did he say about ketanji brown jackson?")

print("\n----------\n")
print(response)

print("\n----------\n")
