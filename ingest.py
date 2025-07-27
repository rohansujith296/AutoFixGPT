# ingest.py

import os
import pdfplumber
import pandas as pd
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.schema import Document


# ✅ Step 1: Convert PDF into .txt
def extract_pdf_to_text(pdf_path, output_txt_path):
    print(f"🔍 Extracting PDF: {pdf_path}")
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])
                all_tables.append(df)

    if not all_tables:
        raise ValueError("❌ No tables found in the PDF!")

    final_df = pd.concat(all_tables, ignore_index=True)

    # Save as .txt for ingestion
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        for _, row in final_df.iterrows():
            f.write(" - ".join(str(cell) for cell in row if pd.notnull(cell)) + "\n")

    print(f"✅ Text file created at: {output_txt_path}")


# ✅ Step 2: Load .txt file and build vector store
def build_vector_store(txt_folder):
    print("📥 Loading and splitting documents...")
    docs = []

    for file in Path(txt_folder).rglob("*.txt"):
        loader = TextLoader(str(file))
        docs.extend(loader.load())

    if not docs:
        raise ValueError("❌ No .txt documents found to embed!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    print(f"🧠 Split into {len(split_docs)} chunks")

    
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local("vector_store")

    print("✅ Vector store saved to 'vector_store/'")

def build_price_vectorstore(csv_path: str):
    print("📥 Loading price dataset CSV...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path, encoding='iso-8859-1')  # or 'cp1252'


    required_cols = {"car", "model", "body part"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"❌ CSV must contain {required_cols}")

    docs = []

    for idx, row in df.iterrows():
        car = str(row["car"])
        model = str(row["model"])
        body_part = str(row["body part"])

        # Collect non-null prices from all shop columns
        shop_prices = []
        for col in df.columns[3:]:
            price = row[col]
            if pd.notnull(price):
                shop_prices.append(f"{col}: ₹{price}")

        # Create a sentence from all the data
        if shop_prices:
            price_info = "; ".join(shop_prices)
            text = f"{car} {model} {body_part} repair cost → {price_info}."
            docs.append(Document(page_content=text))

    print(f"🧠 Created {len(docs)} repair entries")

    # Embed and save
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    

    from langchain_community.embeddings import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local("price_vector_store")

    print("✅ Vector store saved to 'price_vector_store/'")

if __name__ == "__main__":
    # Run extraction and ingestion
    pdf_path = "/Users/rohansujith/Desktop/Python/autofix_gpt/docs/defcodes_database.pdf"         # Change if your PDF is in another folder
    output_txt_path = "docs/dtc_extracted.txt"

    extract_pdf_to_text(pdf_path, output_txt_path)
    build_vector_store("docs")
    csv_file_path = "/Users/rohansujith/Desktop/Python/autofix_gpt/docs/prices_dataset.csv"
    build_price_vectorstore(csv_file_path)
