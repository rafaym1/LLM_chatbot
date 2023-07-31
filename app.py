import os
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from  langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

import json
from typing import Iterable

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm=HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.5, "max_length":500})

# docs=load_docs_from_jsonl('data.jsonl')

embeddings = HuggingFaceEmbeddings()

db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
result = qa({'question': "what is the fee?", 'chat_history': []})

print(result["answer"])