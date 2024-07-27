import os
import jsonlines

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
pinecone_key = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=pinecone_key)
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_key)


class PineconeDocument(BaseModel):
    id: str
    values: list[float]
    metadata: dict = {}

    @classmethod
    def from_document_with_embedding(cls, doc: Document, embedding: list[float]):
        # add the page_content to the metadata
        doc.metadata["page_content"] = doc.page_content
        return cls(id=doc.metadata["id"], values=embedding, metadata=doc.metadata)

    def dict(self):
        return {
            "id": self.id,
            "values": self.values,
            "metadata": self.metadata
        }


def save_documents_to_jsonl(documents: list[PineconeDocument], file_path):
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            doc_dict = doc.dict()
            # to save space, remove the vector
            del doc_dict["values"]
            writer.write(doc_dict)


def get_all_documents() -> list[Document]:
    pages_path = "../pages/"

    excluded_files = ["../pages/index.mdx", "../pages/_app.mdx"]

    documents = []
    for root, dirs, files in os.walk(pages_path):
        for file in files:
            # join in unix style
            file_path = os.path.join(root, file).replace("\\", "/")
            if file_path in excluded_files:
                continue
            if file_path.endswith(".mdx"):
                file_content = open(file_path, "r", encoding="utf8").read()

                # parse to langchain document
                doc = Document(page_content=file_content,
                               metadata={"path": file_path})
                documents.append(doc)

    return documents


def preprocess(document: Document) -> Document:
    # let's see how good it works if we do nothing
    return document


def chunk_pages(documents: list[Document]) -> list[Document]:
    header_level = 3
    chunk_size = 2048
    chunk_overlap = 256

    headers_to_split_on = [(f"{'#' * header_level}", f"Header {header_level}")
                           for header_level in range(1, header_level + 1)]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = []
    for doc in documents:
        doc_id = doc.metadata["path"]
        doc_position = 0
        doc_splits = markdown_splitter.split_text(doc.page_content)
        for split in doc_splits:
            # append metadata without overwriting
            split.metadata.update(doc.metadata)

        # if there is no header 1 then it is just imports so the split can be removed
        doc_splits = [split for split in doc_splits if split.metadata.get(
            "Header 1") is not None]

        for split in doc_splits:
            split_chunks_content = text_splitter.split_text(split.page_content)
            for chunk_content in split_chunks_content:
                chunk = Document(page_content=chunk_content,
                                 metadata=split.metadata.copy())
                doc_position += 1
                chunk.metadata["id"] = f"{doc_id}#{doc_position}"
                chunks.append(chunk)

    print(f"Split to {len(chunks)} chunks")
    return chunks


def setup_index(index_name: str):
    if index_name in pc.list_indexes().names():
        print(f"Index {index_name} already exists. Deleting it.")
        pc.delete_index(name=index_name)

    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

    # disable deletion protection to be able to reset the index
    pc.configure_index(
        name=index_name,
        deletion_protection="disabled",
    )

    index = pc.Index(index_name)
    return index


def upsert_chunks_to_index(index, chunks: list[PineconeDocument], batch_size: int = 256):
    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting chunks to index"):
        batch = chunks[i:i + batch_size]
        index.upsert(vectors=[doc.dict() for doc in batch])


def embed_documents(documents: list[Document], batch_size: int = 256, model: str = "text-embedding-3-small") -> list[tuple[Document, list[float]]]:
    embedded_documents = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding documents"):
        batch = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch]
        response = client.embeddings.create(
            input=batch_texts, model=model).data
        embeddings = [result.embedding for result in response]
        embedded_documents.extend(zip(batch, embeddings))

    return embedded_documents


if __name__ == "__main__":
    index = setup_index(index_name="digital-garden")

    documents = get_all_documents()

    processed_documents = [preprocess(document) for document in documents]

    chunks = chunk_pages(processed_documents)

    embedded_chunks = embed_documents(chunks)
    pinecone_chunks = [PineconeDocument.from_document_with_embedding(
        doc, embedding) for doc, embedding in embedded_chunks]

    upsert_chunks_to_index(index, pinecone_chunks)

    save_documents_to_jsonl(pinecone_chunks, "./chunks.jsonl")
    print("Chunks saved to ./chunks.jsonl")

    # just for fun, how many words have I roughly written
    total_chars = sum([len(chunk.page_content) for chunk in chunks])
    total_words = sum([len(chunk.page_content.split())
                       for chunk in chunks])
    total_book_pages = total_chars // 2000
    print(
        f"You have written about {total_chars} characters, {total_words} words, which is about {total_book_pages} book pages")
