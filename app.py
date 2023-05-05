import json
import gradio as gr
from pathlib import Path
import os
import pandas as pd
import openai
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
import textwrap

# Procesar datos de PDF
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

#import gradio as gr
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity


# API KEY OPENAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class ChatBotInmobiliaria():
    def __init__(self):
        self.embedding_engine = "text-embedding-ada-002"
        self.model_name = "gpt-3.5-turbo"
        self.index = None
    
    def create_dataset(self, directory_path, filepath_dataset):
        # directory_path: Directorio donde se ubican los archivos PDF.
        # filepath_dataset: Nombre del archivo JSON vectorizado.
        if directory_path != None:
            #Leer los PDFs
            pdf = SimpleDirectoryReader(directory_path).load_data()
            #Definir e instanciar el modelo
            modelo = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=self.model_name))
            #Indexar el contenido de los PDFs
            service_context = ServiceContext.from_defaults(llm_predictor=modelo)
            self.index = GPTSimpleVectorIndex.from_documents(pdf, service_context = service_context)
            self.__save_model(filepath_dataset)

    def __save_model(self, filepath):
        #Guardar el índice a disco para no tener que repetir cada vez
        #Recordar que necesistaríamos persistir el drive para que lo mantenga
        self.index.save_to_disk(filepath)
    
    def load_dataset(self, filepath):
        #Cargar el índice del disco
        self.index = GPTSimpleVectorIndex.load_from_disk(filepath)

    def ask(self, question=""):
        if len(question) == 0:
            print("Debe de ingresar una pregunta.")
        try:
            return self.index.query(question)
        except Exception as e:
            print(e)
            return "Hubo un error."

def ask(dataset, pregunta):
    if dataset is None:
        return ""
    path_file = dataset.name
    print(f"Nombre del archivo: {path_file}")
    extension = os.path.splitext(path_file)[1]
    print(f"Extensión el archivo: {extension}")
    dir_name = str(Path(path_file).parent)
    print(f"Carpeta donde está ubicado el archivo: {dir_name}")
    
    if extension.lower() == ".pdf":
        chatbot = ChatBotInmobiliaria()
        DATASET_JSON = "dataset_file.json"
        chatbot.create_dataset(dir_name, DATASET_JSON)
        chatbot.load_dataset(DATASET_JSON)
        return chatbot.ask(question=pregunta)
    elif extension.lower() == ".json":
        chatbot = ChatBotInmobiliaria()
        chatbot.load_dataset(path_file)
        return chatbot.ask(question=pregunta)
    
    

# Gradio

description ="""
<p>
<center>
Demo Inmobiliaria, el objetivo es responder preguntas a través de OpenAI previamente entrenado con un archivo PDF.
<img src="https://raw.githubusercontent.com/All-Aideas/sea_apirest/main/logo.png" alt="logo" width="250"/>
</center>
</p>
"""

article = "<p style='text-align: center'><a href='http://allaideas.com/index.html' target='_blank'>Demo Inmobiliaria: Link para más info</a> </p>"

in1 = gr.inputs.File(label="Archivo PDF")
in2 = gr.inputs.Textbox(label="Pregunta")
out1 = gr.outputs.Textbox(label="Respuesta")

examples = [["demo-inmobiliaria.json", "¿Qué regulaciones tengo para comprar una vivienda?"]]

demo = gr.Interface(
    fn=ask,
    inputs=[in1, in2],
    outputs=out1,
    title="Demo Inmobiliaria",
    description=description,
    article=article,
    enable_queue=True,
    examples=examples,
    )

demo.launch(debug=True)
