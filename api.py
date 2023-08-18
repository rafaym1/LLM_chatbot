from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain



import gradio as gr
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_main():
    return {"message": "This is GIKI's Admission Chatbot"}



import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = 

llm=HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.5, "max_length":500})


def create_chain():

    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()

    from langchain.vectorstores import Chroma
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )


from collections import defaultdict

chain = create_chain()
chat_history = defaultdict(list)



def generate_reply(input_text):


    Userid = 12345
    result = chain({'question': input_text, 'chat_history': chat_history[Userid]})
    chat_history[Userid].append((input_text, result['answer']))
    file1 = open("{0}.txt".format(Userid), "a")  # append mode
    file1.write(input_text + " " + result['answer']+"\n")
    file1.close()
    return result['answer']

# Gradio Interface
iface = gr.Interface(
    fn=generate_reply,  # Function to generate the reply
    inputs=gr.Textbox(),  # Text input component
    outputs=gr.Textbox()  # Text output component
)

app = gr.mount_gradio_app(app, iface, path="/gradio")
