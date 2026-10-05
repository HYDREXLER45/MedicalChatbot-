from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone
import os 
from src.helper import load_pdf_files,filter_to_minimal_docs,text_splitter,download_embeddings
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extract_text = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extract_text)
text_chunks = text_splitter(minimal_docs)
embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name = index_name
)