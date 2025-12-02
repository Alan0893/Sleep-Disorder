# Fix Python Version Issue on Render

## Problem
Render is using Python 3.13 by default, but pandas 2.1.4 doesn't support it.

## Solution 1: Use Updated Requirements (Easiest)
The `requirements.txt` has been updated to use `pandas>=2.2.0` which supports Python 3.13.
Just redeploy and it should work.

## Solution 2: Force Python 3.11 in Render Dashboard

1. Go to your Render service dashboard
2. Click on **Settings** tab
3. Scroll down to **Environment Variables**
4. Click **Add Environment Variable**
5. Add:
   - **Key**: `PYTHON_VERSION`
   - **Value**: `3.11.7`
6. Click **Save Changes**
7. Go to **Manual Deploy** → **Deploy latest commit**

## Solution 3: Use render.yaml (if not already)

If you're using `render.yaml`, make sure it's in your repository root (not in App folder if App is your root directory).

The render.yaml should have:
```yaml
envVars:
  - key: PYTHON_VERSION
    value: 3.11.7
```

## Verify Python Version

After deployment, check the build logs to confirm which Python version is being used.
Look for a line like: "Using Python 3.11.7" or "Python 3.13.4"

