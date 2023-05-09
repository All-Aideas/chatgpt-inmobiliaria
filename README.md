# ChatGPT Inmobiliaria
ChatGPT entrenada con datos de inmobiliaria a partir de archivos en formato PDF. Los archivos PDF deben de contener texto, el cual será utilizado para entrenar la inteligencia artificial. El archivo debe de tener texto que permita al chatbot responder preguntas realizadas por un usuario.

# Configuración
## 1) Carpeta para archivos
Crear una carpeta con el nombre `/dataset`. Esta carpeta almacenará los archivos PDF que serán utilizados para el entrenamiento.

## 2) Configurar API KEY

La variable de entorno `OPENAI_API_KEY` deberá almacenar el API KEY de OpenAI para utilizar ChatGPT. Considerar el siguiente link [clik](https://platform.openai.com/account/api-keys) para su creación. Solicitar permiso de api key a César Riat.

```
os.environ["OPENAI_API_KEY"] = "valor_del_api_key"
```

# Entramiento
## 1)Base de datos:
Se encuentra en Google Drive, se recomienda que ustedes hagan una copia en su perfil, el formato esta en .pdf ya que asi se espera en el Google Colab para poder trabajar, prestar mucha atención al orden de los archivos en las carpetas, la base de datos se ira actualizando [clik](https://drive.google.com/drive/folders/1G8tJ7J7uNS7jm-ls8AY2LUAJW5V87I_U?usp=share_link)
## 2)Entrenamiento en la Nube
Se decidió usar Google Colab, debido a que porrpociona una maquina virtual gratuita por unas horas, subimos los archivos PDF o pueden elegir algún archivo de su elección, eingresar directamente a nuestro colab. Prestar Atención!!! tienen que desacargar el archivo de la base de datos y subirlo a su cuenta del Google Drive [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1-cXaL6EGK-qLDdXEyuuqYPZRsEnHrHB-/view?usp=share_link). Luego tiene que subir el archivo PDF al google drive dentro de la carpeta `/dataset`. Solamente los archivos dentro de esta carpeta serán considerados dentro del entrenamiento.
Luego la red entrenada genera una archivo `demo-inmobiliaria.json`, hay que descargar el mismo.

# Deploy
## 1) Google Colab:
Para hacer un puesta en producción rápido ofrecemos un[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1-cXaL6EGK-qLDdXEyuuqYPZRsEnHrHB-/view?usp=share_link), previamente hay que subir la red entrenada, que obtuvimos en el paso anterior que se llama `demo-inmobiliaria.json`.
Nota; este producto solo durará unas horas ya que esta limitado por el uso de Google Colab.
## 2) HugginFace:
Si se desea tener un modelo en la nube de manera permanente y gratuita le ofrecemos una versión en HuggingFace, el código se encuentra libre en la misma plataforma [Link Deploy](https://huggingface.co/spaces/AllAideas/demo-inmobiliaria)

