# -*- coding: utf-8 -*-
"""
Dog vs Cat Classifier
Adapted for Windows + VS Code + DirectML GPU
"""

import os
import random
import warnings
import zipfile
import urllib.request
import PIL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

warnings.filterwarnings('ignore')

# ==== GPU SETUP ====
print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Usando GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("Nenhuma GPU encontrada, rodando na CPU.")

# ==== DOWNLOAD DO DATASET ====
url = "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"
zip_path = "kagglecatsanddogs_5340.zip"

if not os.path.exists(zip_path):
    print("Baixando dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download concluído.")
else:
    print("Arquivo zip já existe, pulando download.")

# ==== DESCOMPACTAR ====
extract_dir = "data"
if not os.path.exists(os.path.join(extract_dir, "PetImages")):
    print("Extraindo arquivos...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extração concluída.")
else:
    print("Pasta já existe, pulando extração.")

# ==== CRIAR DATAFRAME JÁ FILTRANDO ARQUIVOS INVÁLIDOS ====
input_path = []
label = []

base_path = os.path.join(extract_dir, "PetImages")

for class_name in os.listdir(base_path):
    class_dir = os.path.join(base_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        # Filtro 1: ignora arquivos que não são .jpg
        if not file.lower().endswith('.jpg'):
            continue

        full_path = os.path.join(class_dir, file)

        # Filtro 2: ignora imagens corrompidas
        try:
            img = PIL.Image.open(full_path)
            img.verify()
        except Exception:
            continue

        label.append(1 if class_name == 'Dog' else 0)
        input_path.append(full_path)

print(f"Total de imagens válidas: {len(input_path)}")

df = pd.DataFrame({'images': input_path, 'label': label})
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

# ==== EDA ====
plt.figure(figsize=(25, 25))
temp = df[df['label'] == 1]['images']
start = random.randint(0, len(temp) - 25)
files = temp[start:start + 25]

for index, file in enumerate(files):
    plt.subplot(5, 5, index + 1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('Dogs')
    plt.axis('off')

plt.figure(figsize=(25, 25))
temp = df[df['label'] == 0]['images']
start = random.randint(0, len(temp) - 25)
files = temp[start:start + 25]

for index, file in enumerate(files):
    plt.subplot(5, 5, index + 1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('Cats')
    plt.axis('off')

df['label'] = df['label'].astype('str')

plt.figure()
sns.countplot(x=df['label'])
plt.title("Distribuição de classes (0=Cat, 1=Dog)")
plt.show()

# ==== DATA GENERATORS ====
train, test = train_test_split(df, test_size=0.2, random_state=42)

train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

val_generator = ImageDataGenerator(rescale=1./255)

batch_size = 32  # seguro para rodar localmente

train_iterator = train_generator.flow_from_dataframe(
    train,
    x_col='images',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

val_iterator = val_generator.flow_from_dataframe(
    test,
    x_col='images',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# ==== MODELO ====
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPool2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_iterator, epochs=20, validation_data=val_iterator)

# Salva o modelo treinado em disco
model.save("dog_cat_model.h5")
print("Modelo salvo em dog_cat_model.h5")

# ==== GRÁFICOS ====
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(len(acc))

plt.figure()
plt.plot(epochs_range, acc, 'b', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(epochs_range, loss, 'b', label='Training Loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# ==== TESTE COM IMAGEM REAL ====
image_path = "test.jpg"
if os.path.exists(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)
    pred = model.predict(img)
    result = 'Dog' if pred[0] > 0.5 else 'Cat'
    print("Predição:", result)
else:
    print("Arquivo test.jpg não encontrado, pulando teste final.")