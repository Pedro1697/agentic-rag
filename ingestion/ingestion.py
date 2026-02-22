from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls ]

#each item here is going to be a sublist and each item in that sublist is going to be the document tht we want
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250,
    chunk_overlap = 50
)

doc_splits = text_splitter.split_documents(docs_list)

#vectorestore = Chroma.from_documents(
#    documents=doc_splits,
#    collection_name='rag-chorma',
#    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
#    persist_directory="./.chroma",
#)

retriever = Chroma(
     collection_name='rag-chorma',
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="./.chroma"
).as_retriever()
