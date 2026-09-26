# Deployment Guide — Indian Railways Maintenance AI System

This document provides step-by-step instructions for deploying the project to popular free and cloud hosting platforms (**Render**, **Railway**, **Hugging Face Spaces**, **Docker**, and **Vercel**).

---

## 🌟 Option 1: Render.com (Recommended Free Deployment)

1. **Push your code to GitHub**:
   Ensure your latest code is pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Add production deployment configurations"
   git push origin main
   ```

2. **Create a New Web Service on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
   - Connect your GitHub repository (`Sih_Railway_project`).
   - Configure the following settings:
     - **Name**: `railway-maintenance-ai`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn api:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120`

3. **Set Environment Variables**:
   - Under **Environment Variables**, add:
     - `GEMINI_API_KEY`: `your_actual_gemini_api_key`

4. **Deploy**:
   - Click **Create Web Service**. Render will automatically build the environment, run the pipeline on startup, and provide a public URL (e.g. `https://railway-maintenance-ai.onrender.com`).

---

## 🚆 Option 2: Railway.app

1. Go to [Railway Dashboard](https://railway.app/).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select `Sih_Railway_project`.
4. Railway will automatically detect the `Procfile` and `requirements.txt`.
5. Under **Variables**, add:
   - `GEMINI_API_KEY`: `your_actual_gemini_api_key`
6. Click **Deploy**. Under settings, click **Generate Domain** to get your public URL.

---

## 🤗 Option 3: Hugging Face Spaces (Docker)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Docker** as the Space SDK.
3. Clone your Space repo locally or push your files including `Dockerfile` and `requirements.txt`.
4. Add `GEMINI_API_KEY` under Space **Settings** -> **Repository Secrets**.
5. Hugging Face will build the Docker container and host your FastAPI app and Web Dashboard.

---

## 🐳 Option 4: Docker Local / VPS Deployment

To run the application using Docker locally or on any Linux VPS:

```bash
# 1. Build and run with Docker Compose
docker compose up --build -d

# 2. Access the Web Dashboard
# http://localhost:8000/
```

---

## 🛠️ Verification After Deployment

Once deployed, visit your live app URL:
- **Web Operations Dashboard**: `https://<your-app-domain>/`
- **Health Check API**: `https://<your-app-domain>/health`
- **Swagger Documentation**: `https://<your-app-domain>/docs`
