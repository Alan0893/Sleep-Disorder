# Deploying to Render

## Quick Start

1. **Push your code to GitHub**
   - Make sure the `App` folder contains:
     - `recommendation_app.py`
     - `requirements.txt`
     - `Procfile`
     - `Sleep_health_and_lifestyle_dataset.csv`
     - `templates/recommendation.html`

2. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Sign up or log in

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your app

4. **Configure Settings**
   - **Name**: `sleep-health-recommendations` (or your preferred name)
   - **Environment**: `Python 3`
   - **Python Version**: `3.11.7` (IMPORTANT: pandas 2.1.4 requires Python 3.11, not 3.13)
   - **Root Directory**: `App` (if your app is in the App folder)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn recommendation_app:app --bind 0.0.0.0:$PORT`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for the build to complete (first deployment takes ~5-10 minutes)
   - Your app will be live at `https://your-app-name.onrender.com`

## Important Notes

- **Free Tier**: Render's free tier may spin down after 15 minutes of inactivity. The first request after spin-down takes ~30 seconds to wake up.
- **File Paths**: The app automatically looks for the CSV file in the App directory, which should work on Render.
- **Port**: The app uses the `PORT` environment variable automatically set by Render.

## Troubleshooting

- **Build fails with Python 3.13 error**: 
  - **Option 1**: In Render dashboard → Settings → Environment, add `PYTHON_VERSION=3.11.7` and redeploy
  - **Option 2**: The requirements.txt now uses pandas 2.2.0+ which supports Python 3.13, so you can use Python 3.13
- **Build fails**: Check that all dependencies in `requirements.txt` are correct
- **App crashes**: Check the logs in Render dashboard for error messages
- **CSV not found**: Ensure `Sleep_health_and_lifestyle_dataset.csv` is in the App directory and committed to Git

## Alternative: Using render.yaml

If you prefer, you can use the `render.yaml` file:
1. Push `render.yaml` to your repository
2. In Render dashboard, select "Apply render.yaml" when creating the service
3. Render will automatically use the configuration from the file

