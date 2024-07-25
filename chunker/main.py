import os
import jsonlines
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


def save_documents_to_jsonl(documents: list[Document], file_path):
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.dict())


def load_documents_from_jsonl(file_path) -> list[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents


def get_documents() -> list[Document]:
    pages_path = "../pages/"

    excluded_files = ["../pages/index.mdx", "../pages/_app.mdx"]

    documents = []
    for root, dirs, files in os.walk(pages_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path in excluded_files:
                continue
            if file_path.endswith(".mdx"):
                file_content = open(file_path, "r").read()

                # parse to langchain document
                doc = Document(page_content=file_content,
                               metadata={"path": file_path})
                documents.append(doc)

    return documents


def preprocess(document: Document) -> Document:
    # let's see how good it works if we do nothing
    return document


def chunk_pages() -> list[Document]:
    documents = get_documents()
    print(f"Found {len(documents)} documents")
    processed_documents = [preprocess(document) for document in documents]

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

    md_header_splits = []
    for doc in processed_documents:
        doc_splits = markdown_splitter.split_text(doc.page_content)
        for split in doc_splits:
            # append metadata without overwriting
            split.metadata.update(doc.metadata)

        md_header_splits.extend(doc_splits)

    # if there is no header 1 then it is just imports so the split can be removed
    md_header_splits = [split for split in md_header_splits if split.metadata.get(
        "Header 1") is not None]

    print(f"Split to {len(md_header_splits)} documents on headers")
    chunks = text_splitter.split_documents(md_header_splits)
    print(f"Split to {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    chunks = chunk_pages()

    save_documents_to_jsonl(chunks, "./chunks.jsonl")
    print("Chunks saved to ./chunks.jsonl")

    # just for fun, how many words have I roughly written
    total_chars = sum([len(chunk.page_content) for chunk in chunks])
    total_words = sum([len(chunk.page_content.split()) for chunk in chunks])
    total_book_pages = total_chars // 2000
    print(
        f"You have written about {total_chars} characters, {total_words} words, which is about {total_book_pages} book pages")
