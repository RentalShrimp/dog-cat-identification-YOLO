# -*- coding: utf-8 -*-
"""
Detecção em tempo real de gatos e cachorros na tela
usando YOLOv8 + captura de tela com mss
Pressione ESC para fechar
"""

import numpy as np
import cv2
from mss import mss
from ultralytics import YOLO

# ==== CONFIGURAÇÕES ====
CONFIDENCE = 0.65       # confiança mínima para exibir detecção (50%)
SCALE_DOWN = 0.75     # reduz resolução capturada para processar mais rápido

# Classes do YOLO que nos interessam (dataset COCO)
# 15 = cat, 16 = dog
CLASSES_ALVO = {15: 'Cat', 16: 'Dog'}
CORES = {
    'Cat': (255, 100,   0),   # azul
    'Dog': (0, 120, 255),   # laranja
}

# ==== CARREGA O MODELO YOLO ====
# Na primeira vez baixa automaticamente (~6 MB)
print("Carregando YOLOv8...")
model = YOLO("yolov8n.pt")   # 'n' = nano, o mais leve e rápido
print("Modelo carregado! Pressione ESC para fechar.")

# ==== LOOP DE CAPTURA ====
sct = mss()

while True:
    # Captura monitor principal
    screenshot = sct.grab(sct.monitors[1])
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Reduz para processar mais rápido
    small = cv2.resize(frame, (0, 0), fx=SCALE_DOWN, fy=SCALE_DOWN)

    # Roda o YOLOv8
    results = model(small, verbose=False, conf=CONFIDENCE)[0]

    # Percorre as detecções
    for box in results.boxes:
        cls_id = int(box.cls[0])

        # Ignora classes que não são gato ou cachorro
        if cls_id not in CLASSES_ALVO:
            continue

        label = CLASSES_ALVO[cls_id]
        conf = float(box.conf[0])
        color = CORES[label]

        # Coordenadas em escala reduzida → escala original
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(x1 / SCALE_DOWN)
        y1 = int(y1 / SCALE_DOWN)
        x2 = int(x2 / SCALE_DOWN)
        y2 = int(y2 / SCALE_DOWN)

        # Desenha o quadrado
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Escreve o label + confiança
        texto = f"{label} {conf*100:.0f}%"
        cv2.putText(frame, texto, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Exibe a janela (60% do tamanho para não ocupar a tela toda)
    display = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    cv2.imshow("YOLO Dog & Cat Detector - ESC para sair", display)

    # ESC fecha
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print("Detector encerrado.")
