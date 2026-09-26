# Deployment Guide — Indian Railways Maintenance AI System

This document provides step-by-step instructions for deploying the project to popular free and cloud hosting platforms (**Vercel**, **Render**, **Railway**, **Hugging Face Spaces**, and **Docker**).

---

## 📐 Option 1: Vercel Deployment (Serverless)

1. **Push your latest code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Vercel deployment configuration"
   git push origin main
   ```

2. **Deploy on Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New...** -> **Project**.
   - Import your GitHub repository (`Sih_Railway_project`).
   - Vercel will automatically detect `vercel.json` and use the `@vercel/python` builder for `api.py`.

3. **Set Environment Variables**:
   - Under **Environment Variables**, add:
     - `GEMINI_API_KEY`: `your_actual_gemini_api_key`

4. **Deploy**:
   - Click **Deploy**. Vercel will deploy your FastAPI app and static Operations Web Dashboard serverlessly on a global edge CDN URL (e.g. `https://sih-railway-project.vercel.app`).

---

## 🌟 Option 2: Render.com (Web Service)

1. Go to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Web Service**.
2. Select your repository (`Sih_Railway_project`).
3. Render automatically picks up `render.yaml` and `Procfile`.
4. Add `GEMINI_API_KEY` under Environment Variables and click **Create Web Service**.

---

## 🚆 Option 3: Railway.app

1. Go to [Railway Dashboard](https://railway.app/).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Add `GEMINI_API_KEY` under **Variables** and generate your public domain.

---

## 🤗 Option 4: Hugging Face Spaces (Docker)

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with **Docker** SDK.
2. Push repository code including `Dockerfile` and `requirements.txt`.
3. Add `GEMINI_API_KEY` under Space Repository Secrets.

---

## 🐳 Option 5: Docker Container Deployment

```bash
# Build and run with Docker Compose
docker compose up --build -d
```

---

## 🛠️ Verification After Deployment

Once deployed, visit your live URL:
- **Web Operations Dashboard**: `https://<your-app-domain>/`
- **Health Check API**: `https://<your-app-domain>/health`
- **Swagger Documentation**: `https://<your-app-domain>/docs`
