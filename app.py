import json
import gradio as gr
from pathlib import Path
import os
import pandas as pd
import time
import random
from datetime import datetime
import pytz

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

from pymongo import MongoClient


# API KEY OPENAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# DATABASE CONNECTION
CONNECTION = os.getenv("CONNECTION")
DATABASE = os.getenv("DATABASE")
COLLECTION = os.getenv("COLLECTION")


# CONSTANTES
DATASET_JSON = "demo-inmobiliaria.json"

# Ubicación dataset
carpeta_actual = os.getcwd()
print(f"Nombre de la carpeta actual: {carpeta_actual}")
PATH_FILE = f"{os.getcwd()}/{DATASET_JSON}"
print(f"Ubicación del archivo: {PATH_FILE}")


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
            return self.index.query(f"""
            Actúa como un representante de ventas de una inmobiliaria.
            
            {question} 
            
            Responde en español
            """)
        except Exception as e:
            print(e)
            return "Hubo un error."
    

# Gradio
title = """
<p><center><h1>Demo Inmobiliaria</h1></p></center>
"""

description ="""
<p>
<center>
Demo Inmobiliaria, el objetivo es responder preguntas a través de OpenAI previamente entrenado con un archivo PDF.
<img src="https://raw.githubusercontent.com/All-Aideas/sea_apirest/main/logo.png" alt="logo" width="250"/>
</center>
</p>
"""

article = "<p style='text-align: center'><a href='http://allaideas.com/index.html' target='_blank'>Demo Inmobiliaria: Link para más info</a> </p>"
examples = [["¿Cuánto está una casa en San Isidro?"],["Hay precios más baratos?"],["A dónde llamo?"],["Qué leyes existen?"]]

gpt_bot = ChatBotInmobiliaria()
gpt_bot.load_dataset(PATH_FILE)


# Conexión a la base de datos MongoDB
client = MongoClient(CONNECTION)
db = client[DATABASE]
collection = db[COLLECTION]


def get_datetime():
    # Obtener la hora actual
    hora_actual = datetime.now()
    # Obtener la zona horaria de la hora actual
    zona_horaria_actual = pytz.timezone('America/Argentina/Buenos_Aires')
    # Aplicar la zona horaria a la hora actual
    hora_actual_con_zona_horaria = hora_actual.astimezone(zona_horaria_actual)
    return hora_actual_con_zona_horaria


def insert_chat(data):
    return collection.insert_one({"conversacion": data})


def update_chat(id, data):
    collection.update_one({"_id": id}, {"$set": {"conversacion": data}})


def add_chat_history(chat_history, message, answer, calificacion=None):
    global json_chat_history
    global id_chat

    json_chat = {"message": message, 
                 "answer": answer, 
                 "datetime": get_datetime(), 
                 "calificacion": calificacion}
    if len(chat_history) > 0:
        # Si chat_history no está vacía, significa que es una continuación de la conversación anterior
        json_chat_history.append(json_chat)
        # chat_history.append([message, answer])

        update_chat(id_chat, json_chat_history)
    else:
        # Si chat_history está vacía, es una nueva conversación
        json_chat_history = []
        json_chat_history.append(json_chat)
        # chat_history.append([message, answer])

        # Almacenar la nueva conversación en la base de datos
        db_result = insert_chat(json_chat_history)
        id_chat = db_result.inserted_id


with gr.Blocks() as demo:    
    gr.Markdown(f"""
    {title}
    {description}
    """)

    out1 = gr.Chatbot(label="Respuesta").style(height=300)
    
    with gr.Row():
        in2 = gr.Textbox(label="Pregunta")
        enter = gr.Button("Enviar mensaje")
    
    with gr.Row():
        upvote_btn = gr.Button(value="👍  Conforme", interactive=True)
        downvote_btn = gr.Button(value="👎  No conforme", interactive=True)
        flag_btn = gr.Button(value="⚠️  Alertar", interactive=True)
        # regenerate_btn = gr.Button(value="🔄  Regenerar", interactive=False)
    clear_btn = gr.Button(value="🗑️  Nuevo chat", interactive=True)
    
    gr.Markdown(article)


    def respond(message, chat_history):
        answer = str(gpt_bot.ask(question=message))
        add_chat_history(chat_history=chat_history, 
                    message=message, 
                    answer=answer)
        chat_history.append([message, answer])
        time.sleep(1)
        return "", chat_history


    enter.click(fn=respond, inputs=[in2, out1], outputs=[in2, out1])
    in2.submit(respond, [in2, out1], [in2, out1])


    def upvote_last_response(message, chat_history):
        """
        Obtener el último objeto JSON de la lista
        Actualizar el valor del atributo "calificacion"
        """
        if len(json_chat_history) > 0:
            json_chat_history[-1]["calificacion"] = "Conforme"
            update_chat(id_chat, json_chat_history)
        
        return message, chat_history


     def downvote_last_response(message, chat_history):
         """
         Obtener el último objeto JSON de la lista
         Actualizar el valor del atributo "calificacion"
         """
         if len(json_chat_history) > 0:
             json_chat_history[-1]["calificacion"] = "No conforme"
             update_chat(id_chat, json_chat_history)
        
         return message, chat_history


    def flag_last_response(message, chat_history):
        """
        Obtener el último objeto JSON de la lista
        Actualizar el valor del atributo "calificacion"
        """
        if len(json_chat_history) > 0:
            json_chat_history[-1]["calificacion"] = "Alertar"
            update_chat(id_chat, json_chat_history)
        
        return message, chat_history


    # def regenerate_answer(message, chat_history):
    #     """
    #     Obtener el último objeto JSON de la lista
    #     Actualizar el valor del atributo "calificacion"
    #     """
    #     if len(json_chat_history) > 0:
    #         pregunta = json_chat_history[-1]["message"]
    #         answer = str(gpt_bot.ask(question=pregunta))
    #         add_chat_history(chat_history=chat_history, 
    #                          message=pregunta, 
    #                          answer=answer,
    #                          calificacion="Regenerado")
    #         chat_history.pop(-1)
    #         chat_history.append([message, answer])
    #         time.sleep(1)        
    #     return message, chat_history


    upvote_btn.click(upvote_last_response, inputs=[in2, out1], outputs=[in2, out1])
    downvote_btn.click(downvote_last_response, inputs=[in2, out1], outputs=[in2, out1])
    flag_btn.click(flag_last_response, inputs=[in2, out1], outputs=[in2, out1])
    # regenerate_btn.click(regenerate_answer, inputs=[in2, out1], outputs=[in2, out1])
    clear_btn.click(lambda: None, None, out1, queue=False)

    
# in1 = gr.inputs.Textbox(label="Pregunta")
# out1 = gr.outputs.Chatbot(label="Respuesta").style(height=350)

# demo = gr.Interface(
#     fn=chat,
#     inputs=in1,
#     outputs=out1,
#     title="Demo Inmobiliaria",
#     description=description,
#     article=article,
#     enable_queue=True,
#     examples=examples,
#     )

demo.launch(debug=True)
