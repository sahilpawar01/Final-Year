# Image Captioning Application

A Flask web application that generates captions for images using a deep learning model based on InceptionV3 and LSTM.

## Features

- Upload images (PNG, JPG, JPEG)
- Automatic caption generation using beam search
- Modern, responsive UI with Bootstrap

## Local Setup

1. Create a virtual environment:
```bash
python -m venv caption_env
```

2. Activate the virtual environment:
   - Windows: `caption_env\Scripts\activate`
   - Linux/Mac: `source caption_env/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Deployment on Render

### Prerequisites
- GitHub account
- Render account (sign up at https://render.com)

### Step 1: Push to GitHub

1. Initialize git repository (if not already done):
```bash
git init
git remote add origin https://github.com/sahilpawar01/Final-Year.git
```

2. Add all files:
```bash
git add .
```

3. Commit:
```bash
git commit -m "Initial commit - Image Captioning App"
```

4. Push to GitHub:
```bash
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render

#### Option A: Using render.yaml (Recommended)

1. Go to https://dashboard.render.com
2. Click "New +" and select "Blueprint"
3. Connect your GitHub repository: `sahilpawar01/Final-Year`
4. Render will automatically detect the `render.yaml` file
5. Click "Apply" to deploy

#### Option B: Manual Deployment

1. Go to https://dashboard.render.com
2. Click "New +" and select "Web Service"
3. Connect your GitHub account and select the repository: `sahilpawar01/Final-Year`
4. Configure the service:
   - **Name**: image-captioning-app (or any name you prefer)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Select a plan (Starter plan is free with limitations)
5. Click "Create Web Service"

### Important Notes for Render Deployment

1. **Model Files**: Make sure `caption_model_safe.h5` and `tokenizer_safe.pkl` are committed to GitHub (they should be if they're in your project root)

2. **File Size Limits**: 
   - Free tier on Render has limitations
   - If your model file is too large (>100MB), consider using Git LFS or hosting the model files separately

3. **Build Time**: The first build may take 10-15 minutes as it downloads TensorFlow and other dependencies

4. **Memory**: TensorFlow models require significant memory. Consider upgrading to a paid plan if you encounter memory issues

5. **Environment Variables**: The app automatically uses the PORT environment variable provided by Render

## Project Structure

```
.
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment configuration
├── caption_model_safe.h5  # Trained caption model
├── tokenizer_safe.pkl    # Tokenizer for text processing
├── templates/
│   └── index.html        # Frontend template
└── static/
    ├── uploads/          # Uploaded images directory
    └── style.css         # Custom styles
```

## Technologies Used

- Flask - Web framework
- TensorFlow/Keras - Deep learning framework
- InceptionV3 - Image feature extraction
- Bootstrap 5 - Frontend styling
- Gunicorn - Production WSGI server

## License

This project is for educational purposes.

