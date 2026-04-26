# 🌿 BlightGuard: Potato Leaf Disease Detection

BlightGuard is an AI-powered web application that detects potato leaf diseases using Deep Learning. It classifies images into **Healthy**, **Early Blight**, and **Late Blight** to help farmers take timely action.

---

## 🚀 Features

* 🌱 Disease classification using CNN
* 🧠 Deep Learning (TensorFlow/Keras)
* 🌐 React web application
* ⚡ FastAPI backend
* ☁️ Optional deployment with TensorFlow Serving & GCP

---

## 🛠 Tech Stack

* **Frontend:** React.js
* **Backend:** FastAPI
* **ML/DL:** TensorFlow, Keras
* **Deployment:** Docker, GCP

---

## ⚙️ Setup Instructions

---

### 🔹 1. Python Setup

Install Python and required packages:

```bash
pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt
```

---

### 🔹 2. React Frontend Setup

```bash
cd frontend
npm install --from-lock-json
npm audit fix
```

Copy environment file:

```bash
cp .env.example .env
```

Update API URL in `.env`:

```env
REACT_APP_API_URL=http://localhost:8000/predict
```

---

## 🧠 Training the Model

1. Download dataset from Kaggle
2. Keep only potato-related folders
3. Run Jupyter Notebook:

```bash
jupyter notebook
```

4. Open:

```
training/potato-disease-training.ipynb
```

5. Update dataset path in cell #2
6. Run all cells
7. Save model in `/models` folder

---

## 🔌 Running the API

### ▶️ FastAPI

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0
```

API runs at:

```
http://localhost:8000
```

---

### ▶️ FastAPI + TensorFlow Serving (Optional)

```bash
cd api
```

Update config:

```
models.config.example → models.config
```

Run TF Serving:

```bash
docker run -t --rm -p 8501:8501 \
-v C:/Code/potato-disease-classification:/potato-disease-classification \
tensorflow/serving \
--rest_api_port=8501 \
--model_config_file=/potato-disease-classification/models.config
```

Run API:

```bash
uvicorn main-tf-serving:app --reload --host 0.0.0.0
```

---

## 🌐 Running the Frontend

```bash
cd frontend
npm start
```

---

## 🔄 TF Lite Model Conversion

```bash
jupyter notebook
```

Open:

```
training/tf-lite-converter.ipynb
```

Run all cells → model saved in `tf-lite-models`

---

## ☁️ Deployment (GCP)

### 🔹 TF Lite Model

```bash
cd gcp
gcloud auth login
gcloud functions deploy predict_lite \
--runtime python38 \
--trigger-http \
--memory 512 \
--project project_id
```

---

### 🔹 TensorFlow (.h5 Model)

```bash
cd gcp
gcloud functions deploy predict \
--runtime python38 \
--trigger-http \
--memory 512 \
--project project_id
```

---

## 📸 Demo

> Add screenshots of your app here

---

## 📌 Project Structure

```
Blightguard/
  │
  ├── Data_Set/
    │
    ├── api/                # FastAPI backend
    ├── frontend/           # React frontend
    ├── training/           # Model training notebooks
    ├── models/             # Saved models
    ├── gcp/                # Deployment scripts
```

---

## ⚠️ Notes

* Do not upload large model files (>100MB) to GitHub
* Use cloud storage (Google Drive / GCP bucket)
* Ensure correct API URL in `.env`

---

## 👨‍💻 Author

**Saud Shabbir Sayyed**


---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share!
