# dog-cat-identification
# Dog vs Cat Classification + Real-Time Detection (YOLOv8 + Screen Capture)

This project implements a complete pipeline for:

1. Training a **Convolutional Neural Network (CNN)** to classify images as **dog** or **cat**.
2. Saving the trained model for later use.
3. Running **single-image predictions**.
4. Running **real-time detection of dogs and cats on your computer screen** using:
   - a simple sliding-window approach with the CNN, and/or  
   - **YOLOv8** (Ultralytics) pre-trained on COCO, combined with screen capture.

The code is adapted to run on **Windows 10/11** with **Python 3.10** and **TensorFlow 2.10** using **DirectML** to leverage the GPU (e.g., NVIDIA GeForce RTX 2060).

---

## 1. Project Structure

Main scripts:

- `dog_cat_id_v0.py`  
  Trains a CNN from scratch on the **Kaggle Cats and Dogs** dataset (Microsoft) and saves the model.

- `predict.py`  
  Loads the trained model (`dog_cat_model.h5`) and predicts whether a given image is a dog or a cat.

- `yolo_screen_detect.py`  
  Real-time dog & cat detection on the screen using **YOLOv8** (Ultralytics) and screen capture (videos, browser, etc.).

Files generated after running:

- `kagglecatsanddogs_5340.zip` – downloaded dataset.
- `data/` – extracted dataset, including `PetImages/Cat` and `PetImages/Dog`.
- `dog_cat_model.h5` – trained Keras model.
- `test.jpg` – image used for simple prediction tests.

---

## 2. Requirements

- **OS:** Windows 10 or 11  
- **Python:** 3.10 (recommended)  
- **GPU (optional but recommended):** any DirectML-compatible GPU (e.g., NVIDIA RTX 2060)  
- **Editor:** VS Code or any other (optional, but used here)

---

## 3. Environment Setup

### 3.1. Create and activate virtual environment

In the project folder:

```bash
cd D:\IA\dog-cat-identification  # or your project path

# Create venv with Python 3.10
py -3.10 -m venv venv_tf

# Activate (PowerShell)
venv_tf\Scripts\activate
```

If PowerShell blocks script execution, run (once):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```bash
venv_tf\Scripts\activate
```

---

## 4. Install Dependencies

With `venv_tf` **activated**, install the following.

### 4.1. Core ML stack (TensorFlow + DirectML)

```bash
pip install "numpy==1.26.4"
pip install "tensorflow-cpu==2.10"
pip install tensorflow-directml-plugin

pip install pandas matplotlib seaborn scikit-learn pillow tqdm
```

Notes:

- `tensorflow-cpu==2.10` + `tensorflow-directml-plugin` allow using the GPU via **DirectML** on Windows.
- `numpy==1.26.4` is required for compatibility with TensorFlow 2.10.

### 4.2. Real-time detection (screen capture + YOLO)

```bash
pip install mss opencv-python
pip install ultralytics
```

---

## 5. Training the CNN (dog vs cat)

The original script came from a Google Colab notebook and was adapted for local execution and Windows paths.

### 5.1. What `dog_cat_id_v0.py` does

1. **Downloads** the Cats & Dogs dataset from Microsoft:

   ```python
   url = "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"
   ```

2. **Extracts** it into `data/PetImages`.

3. **Builds a DataFrame** of image paths and labels:
   - `images`: path to the image file.
   - `label`: `1` for dog, `0` for cat.

4. **Cleans** the dataset:
   - Filters only `.jpg` files.
   - Attempts to open each image with `PIL.Image` and discards corrupted ones.

5. **EDA (Exploratory Data Analysis):**
   - Plots 25 random dog images.
   - Plots 25 random cat images.
   - Displays a count plot of labels to show class balance.

6. **Creates data generators** with `ImageDataGenerator`:
   - Rescaling (`1./255`).
   - Data augmentation: rotation, shear, zoom, horizontal flip, etc.

7. **Splits data** into training and validation with `train_test_split`.

8. **Defines the CNN model** (Keras `Sequential`):

   - Input: `(224, 224, 3)`
   - Layers:
     - `Conv2D(16)` → `MaxPool2D`
     - `Conv2D(32)` → `MaxPool2D`
     - `Conv2D(64)` → `MaxPool2D`
     - `Flatten`
     - `Dense(512, relu)`
     - `Dense(1, sigmoid)` (binary output: dog vs cat)

9. **Compiles** the model:

   ```python
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   ```

10. **Trains** for a given number of epochs (e.g. 10–50) using:

    ```python
    history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)
    ```

11. **Saves the trained model**:

    ```python
    model.save("dog_cat_model.h5")
    ```

12. **Plots metrics**:
    - Training vs validation accuracy.
    - Training vs validation loss.

### 5.2. Run training

```bash
python dog_cat_id_v0.py
```

Expected output (example):

```text
... - loss: 0.36 - accuracy: 0.83 - val_loss: 0.37 - val_accuracy: 0.84
Modelo salvo em dog_cat_model.h5
```

You will now have `dog_cat_model.h5` in your project directory.

---

## 6. Single Image Prediction

Use `predict.py` to classify a single image.

### 6.1. Prepare the test image

Place a JPG image of a dog or cat in the project folder and name it:

```text
D:\IA\dog-cat-identification\test.jpg
```

(or adapt the file name in the script).

### 6.2. How `predict.py` works

1. Loads the model:

   ```python
   model = tf.keras.models.load_model("dog_cat_model.h5")
   ```

2. Loads and preprocesses the image:

   ```python
   img = load_img(image_path, target_size=(224, 224))
   img = np.array(img) / 255.0
   img = img.reshape(1, 224, 224, 3)
   ```

3. Predicts and prints the label:

   ```python
   pred = model.predict(img)
   label = 'Dog' if pred[0] > 0.5 else 'Cat'
   ```

### 6.3. Run prediction

```bash
python predict.py
```

Example output:

```text
Resultado: Dog
Probabilidade: 0.9123
(Closer to 1.0 means dog, closer to 0.0 means cat.)
```

You can easily adapt the script to accept a file path argument (e.g. `python predict.py my_image.jpg`) if desired.

---

## 7. Real-Time Screen Detection (YOLOv8)

This part uses **YOLOv8** (Ultralytics) pre-trained on the COCO dataset and **screen capture** to detect dogs and cats in anything displayed on your monitor (videos, images, web pages, etc.).

### 7.1. Concept

- Capture the screen in a loop with `mss`.
- Downscale the image for performance.
- Run YOLOv8 (`yolov8n.pt`) on the captured frame.
- Filter only COCO classes:
  - `15` = cat
  - `16` = dog
- Draw bounding boxes and class labels on the screen preview.

### 7.2. Key settings in `yolo_screen_detect.py`

```python
CONFIDENCE = 0.65       # minimum confidence threshold (e.g., 0.65)
SCALE_DOWN  = 0.75      # downscale factor for faster processing

# COCO classes of interest
CLASSES_ALVO = {15: 'Cat', 16: 'Dog'}
CORES = {
    'Cat': (255, 100,   0),  # blue-ish
    'Dog': (  0, 120, 255),  # orange-ish
}
```

Main steps:

1. **Load YOLOv8 model:**

   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   ```

   - On first run, this will download the weights automatically.

2. **Capture screen and run inference in a loop:**

   ```python
   from mss import mss
   sct = mss()

   screenshot = sct.grab(sct.monitors[1])  # captures primary monitor
   frame = np.array(screenshot)
   # process with YOLO, draw boxes, etc.
   ```

3. **Exit condition:**  
   Press `ESC` (Esc key) to close the OpenCV window and stop detection.

### 7.3. Run YOLO screen detection

```bash
python yolo_screen_detect.py
```

Recommended way to test:

- Open a dog kennel video / pet video / animal documentary on YouTube.
- Put the video in a visible area of the screen.
- Run the script and watch the detection window show bounding boxes and labels **Dog** / **Cat**.

You can tune:

- `CONFIDENCE` to be stricter (e.g. 0.75) or more permissive (e.g. 0.5).
- `SCALE_DOWN` (smaller values like `0.5` reduce resolution and improve speed).

---

## 8. Notes and Best Practices

- **GPU usage:**  
  With TensorFlow 2.10 on Windows, GPU is enabled through the **DirectML** plugin. It may not reach 100% utilization or match pure CUDA performance, but it’s good enough for this project.

- **Batch size:**  
  The original Colab notebook used `batch_size=512`. On local machines, a safer starting point is `32` or `64`, then adjust according to available VRAM.

- **Dataset size:**  
  The dataset is relatively big (~800 MB). Prefer installing it on an SSD for better performance.

- **Modularity:**  
  Keep:
  - `dog_cat_id_v0.py` → training + saving model  
  - `predict.py`       → single image prediction  
  - `yolo_screen_detect.py` → real-time detection (YOLOv8)  
  - `screen_detect.py` → optional: sliding-window detection with the custom CNN  

- **Next steps:**  
  - Replace the simple CNN with a transfer learning backbone (ResNet, EfficientNet, etc.).  
  - Train a custom YOLO model specifically for cats and dogs.  
  - Use a webcam instead of screen capture.  
  - Build a simple GUI for loading images and running predictions.

---

## 9. License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.
```
