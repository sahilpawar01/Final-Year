# app.py
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CONFIG
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_LENGTH = 34   # <-- set this to the max_length you used in Colab
MODEL_FILENAME = "caption_model_safe.h5"
TOKENIZER_FILENAME = "tokenizer_safe.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load tokenizer
with open(TOKENIZER_FILENAME, "rb") as f:
    tokenizer = pickle.load(f)

# Load trained caption model
model = load_model(MODEL_FILENAME)

# Load InceptionV3 feature extractor (weights auto-download on first run)
base_model = InceptionV3(weights="imagenet")
feature_extractor = Model(base_model.input, base_model.layers[-2].output)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def extract_features(path):
    img = load_img(path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = feature_extractor.predict(x, verbose=0)
    return feat[0]

def beam_search_caption(photo_feature, beam_width=3):
    # Simple beam search consistent with training code
    start = tokenizer.word_index.get("startseq", 1)
    end = tokenizer.word_index.get("endseq", 2)
    sequences = [[ [start], 0.0 ]]

    for _ in range(MAX_LENGTH):
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=MAX_LENGTH)
            preds = model.predict([photo_feature.reshape(1,2048), padded], verbose=0)[0]
            # top k
            top_k = np.argsort(preds)[-beam_width:]
            for w in top_k:
                new_seq = seq + [int(w)]
                new_score = score + np.log(preds[w] + 1e-12)
                all_candidates.append([new_seq, new_score])
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = sequences[0][0]
    words = []
    for w in best_seq:
        word = tokenizer.index_word.get(w)
        if word is None:
            continue
        if word in ("startseq", "endseq"):
            continue
        words.append(word)
    return " ".join(words)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/caption", methods=["POST"])
def caption_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            feat = extract_features(save_path)
            caption = beam_search_caption(feat, beam_width=3)
            return jsonify({"caption": caption, "filename": filename})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed. Use png/jpg/jpeg."}), 400

# optional: serve uploaded file (already available under static/uploads)
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    # Use PORT environment variable if available (for Render), otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Debug mode disabled for production
    app.run(host="0.0.0.0", port=port, debug=False)
