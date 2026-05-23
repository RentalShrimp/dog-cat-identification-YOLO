"""
Teste de predição com modelo já treinado
Coloque a imagem que quer testar como test.jpg na pasta do projeto
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

# ==== CARREGA O MODELO SALVO ====
print("Carregando modelo...")
model = tf.keras.models.load_model("dog_cat_model.h5")
print("Modelo carregado com sucesso!")

# ==== TESTE COM IMAGEM REAL ====
image_path = "teste2.jpg"

img = load_img(image_path, target_size=(224, 224))
img = np.array(img) / 255.0
img = img.reshape(1, 224, 224, 3)

pred = model.predict(img)
probabilidade = float(pred[0])

if probabilidade > 0.5:
    resultado = "Dog"
else:
    resultado = "Cat"

print(f"\nResultado: {resultado}")
print(f"Probabilidade: {probabilidade:.4f}")
print("(Quanto mais perto de 1.0, mais certeza de cachorro. Mais perto de 0.0, mais certeza de gato.)")
