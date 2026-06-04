"""
Here will upload the data and chunk it.
"""
# import library will be used 
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader,TextLoader, CSVLoader ,Docx2txtLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


## read all files fun

def load_all_docs(data_dir:str) -> List[Any]:
    """
    Loaded all the supporter files from the data directory and convert it to LangChain document structure
    supported file: CSV, PDF, EXCELL, WORD and JSON.
    """
    
    # use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path : {data_path}")
    documents =[]

    # pdf files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loaded PDF {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)

        except Exception as e :
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")
    
    ## for text file 
    text_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Founded {len(text_files)} Text files: {[str(t) for t in text_files]} ")

    for text_file in text_files:
        print(f"[DEBUG] Loaded text in {text_file}")
        try:
            loader = TextLoader(str(text_file), encoding="utf-8")
            loaded = loader.load()
            print(f"[debug] loaded {len(loaded)} texts docs from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load TEXT {text_file}: {e} ")
    
    # for CSV file
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"Founded {len(csv_files)} CSV files: {(str(c) for c in csv_files)}")

    for csv_file in csv_files:
        print(f"[DEBUG] loaded csv file: {csv_file}")
        try:
            loader = CSVLoader(csv_file)
            loaded = loader.load()
            print(f"[DEBUG] loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)

        except Exception as e:
            print(f" failed loaded {csv_file}: {e}")
    
    
    return documents




    




    

