from tensorflow.keras import models
import cv2,os,numpy as np

model = models.load_model("cats_dogs.keras")

## fonction de prediction
def predict(img):
    ## Charger l'image
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = cv2.resize(img,(64,64))
    image = np.expand_dims(image, axis=0)
    ## normaliser
    image = image/255.0

    ## resultat
    res = model(image,training=False)
    return {"Chat":float(1-res[0][0]),"Chien":float(res[0][0])}


import gradio as gr
iface = gr.Interface(fn= predict,inputs="image",outputs="text")
iface.launch()

