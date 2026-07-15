import os
import cv2
import numpy as np
from tensorflow.keras import models
import gradio as gr

# Charger le modèle
model = models.load_model("cats_dogs.keras")

## fonction de prediction
def predict(img):
    ## Charger l'image
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64)) 
    image = np.expand_dims(image, axis=0)##model attend des données de type(batch,64,64,canal)
    ## normaliser
    image = image / 255.0

    ## resultat
    res = model(image, training=False)
    # Renvoyer un dictionnaire est parfait avec l'output "label" de Gradio
    return {"Chat": float(1 - res[0][0]), "Chien": float(res[0][0])}

# On utilise "label" en sortie pour afficher un joli graphique des probabilités
iface = gr.Interface(fn=predict, inputs="image", outputs="label")

# Configuration obligatoire pour Render
if __name__ == "__main__":
    # Récupère le port fourni par Render, ou 7860 par défaut en local
    port = int(os.environ.get("PORT", 7860))
    
    # On force Gradio à écouter sur l'IP 0.0.0.0 et sur le bon port
    iface.launch(server_name="0.0.0.0", server_port=port)