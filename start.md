# Deployment Guide

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py          # → http://localhost:8000
                        # API docs → http://localhost:8000/docs

# Frontend (new terminal)
cd frontend
npm install
npm run dev             # → http://localhost:3000
```

---

## Deploy Backend → Render (from GitHub)

1. Push this repo to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — confirm these settings:
   - **Build command:** `pip install -r backend/requirements.txt`
   - **Start command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy**
6. Copy your Render URL: `https://your-app.onrender.com`

---

## Deploy Frontend → Vercel (from GitHub)

1. Go to https://vercel.com → New Project → import your GitHub repo
2. Set **Root Directory** to `frontend`
3. Add environment variable:
   - Key: `BACKEND_URL`
   - Value: `https://your-app.onrender.com`  ← your Render URL
4. Click **Deploy**
5. Copy your Vercel URL: `https://your-app.vercel.app`

---

## After Both Are Deployed

Go to **Render dashboard → your service → Environment** and add:
- Key: `FRONTEND_URL`
- Value: `https://your-app.vercel.app`  ← your Vercel URL

Then click **Manual Deploy** to restart with CORS updated.

---

## Also update vercel.json

Replace the placeholder in `vercel.json`:
```json
"destination": "https://your-render-app.onrender.com/api/:path*"
```
with your actual Render URL, then push to GitHub. Vercel redeploys automatically.
