# Render Deployment Guide

## ✅ Files Ready for Deployment

All necessary files have been created and pushed to GitHub:
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Excludes virtual environment and unnecessary files
- ✅ `render.yaml` - Render deployment configuration
- ✅ `app.py` - Updated for production (uses PORT environment variable)
- ✅ `README.md` - Complete documentation

## 🚀 How to Deploy on Render

### Method 1: Using render.yaml (Easiest)

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Sign up or log in

2. **Create New Blueprint**
   - Click "New +" button
   - Select "Blueprint"

3. **Connect Repository**
   - Connect your GitHub account if not already connected
   - Select repository: `sahilpawar01/Final-Year`
   - Click "Apply"

4. **Wait for Deployment**
   - Render will automatically detect `render.yaml`
   - Build will start automatically
   - First build takes 10-15 minutes (downloading TensorFlow)

### Method 2: Manual Web Service

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Sign up or log in

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"

3. **Connect Repository**
   - Connect your GitHub account
   - Select repository: `sahilpawar01/Final-Year`

4. **Configure Service**
   - **Name**: `image-captioning-app` (or any name)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: (leave empty)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: 
     - **Free**: Limited resources (may have memory issues with TensorFlow)
     - **Starter ($7/month)**: Recommended for TensorFlow models

5. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete (10-15 minutes)

## ⚠️ Important Notes

### File Size Warning
GitHub warned that `caption_model_safe.h5` (64.55 MB) exceeds the recommended 50 MB limit. This is fine for now, but if you encounter issues:

**Option 1: Use Git LFS (Recommended for large files)**
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add caption_model_safe.h5
git commit -m "Add model file with LFS"
git push
```

**Option 2: Host model separately**
- Upload model to cloud storage (AWS S3, Google Cloud Storage)
- Download during build process

### Memory Requirements
- TensorFlow models require significant memory
- Free tier may have limitations
- If you get memory errors, upgrade to Starter plan ($7/month)

### Build Time
- First build: 10-15 minutes (downloading dependencies)
- Subsequent builds: 5-10 minutes

### Environment Variables
The app automatically uses Render's `PORT` environment variable. No manual configuration needed.

## 🔍 Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Ensure all files are in GitHub
- Verify `requirements.txt` has correct versions

### App Crashes After Deployment
- Check logs in Render dashboard
- Verify model files are present
- Check memory usage (may need to upgrade plan)

### Slow Response Times
- TensorFlow model loading takes time
- First request after idle period may be slow (cold start)
- Consider using a paid plan for better performance

## 📝 Next Steps

1. ✅ Repository is on GitHub: https://github.com/sahilpawar01/Final-Year
2. ⏭️ Deploy on Render using one of the methods above
3. ⏭️ Test your deployed application
4. ⏭️ Share your app URL!

## 🎉 Success!

Once deployed, you'll get a URL like: `https://image-captioning-app.onrender.com`

Your app will be live and accessible worldwide!

