# Fix for Render Build Error

## Problem
Render is using Python 3.13, which causes `setuptools.build_meta` import errors because TensorFlow and some dependencies don't fully support Python 3.13 yet.

## Solution: Manual Configuration in Render Dashboard

Since `render.yaml` Blueprint might not properly set Python version, you need to configure it manually:

### Step 1: Delete Current Service (if exists)
1. Go to https://dashboard.render.com
2. Find your service
3. Click on it → Settings → Delete

### Step 2: Create New Web Service (Manual Configuration)

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Connect GitHub if not already connected
   - Select: `sahilpawar01/Final-Year`
   - Branch: `main`

3. **Configure Service - IMPORTANT SETTINGS:**
   - **Name**: `image-captioning-app`
   - **Environment**: `Python 3` 
   - **Python Version**: **Select `3.10.12` or `3.10`** ⚠️ THIS IS CRITICAL
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: (leave empty)
   - **Build Command**: `pip install --upgrade pip && pip install setuptools wheel && pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Starter` ($7/month) - Recommended for TensorFlow

4. **Click "Create Web Service"**

### Step 3: Verify Python Version
After deployment starts, check the build logs. You should see Python 3.10.x being used, not 3.13.

## Alternative: If Python Version Selection Not Available

If you can't select Python version in the UI, try this workaround:

1. Create a `.python-version` file (for pyenv)
2. Or use a `Dockerfile` approach

But the manual configuration method above should work.

## Updated Files
- ✅ `requirements.txt` - Updated setuptools/wheel versions
- ✅ `render.yaml` - Simplified build command
- ✅ `runtime.txt` - Specifies Python 3.10.12

The key is selecting Python 3.10 in the Render dashboard UI during service creation.

