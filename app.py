# app.py
import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import importlib
import types

# Create compatibility shim for old Keras module paths
# This allows pickle to find the old module paths
keras_legacy_module = types.ModuleType('keras.src.legacy.preprocessing.text')
keras_legacy_module_preprocessing = types.ModuleType('keras.src.legacy.preprocessing')
keras_legacy = types.ModuleType('keras.src.legacy')
keras_src = types.ModuleType('keras.src')
keras_module = types.ModuleType('keras')

# Import the actual Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

# Make Tokenizer available through the old paths
keras_legacy_module.Tokenizer = Tokenizer
keras_legacy_module_preprocessing.text = keras_legacy_module
keras_legacy.preprocessing = keras_legacy_module_preprocessing
keras_src.legacy = keras_legacy
keras_module.src = keras_src

# Register the shim modules in sys.modules
sys.modules['keras.src.legacy.preprocessing.text'] = keras_legacy_module
sys.modules['keras.src.legacy.preprocessing'] = keras_legacy_module_preprocessing
sys.modules['keras.src.legacy'] = keras_legacy
sys.modules['keras.src'] = keras_src

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

# Load tokenizer - the shim modules should allow pickle to find the old paths
try:
    with open(TOKENIZER_FILENAME, "rb") as f:
        tokenizer = pickle.load(f, encoding='latin1')
except TypeError:
    # If encoding parameter not supported, try without it
    with open(TOKENIZER_FILENAME, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading tokenizer: {e}", file=sys.stderr)
    raise

# Load trained caption model with compatibility handling
# The model was saved with older Keras that used 'batch_shape' instead of 'input_shape'
from tensorflow.keras.layers import InputLayer as BaseInputLayer, Embedding as BaseEmbedding
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import backend as K
import json

# Monkey-patch the deserialization to fix configs before they reach layer constructors
def fix_layer_config(config):
    """Fix layer configs to handle old Keras formats"""
    config = config.copy()
    
    # Fix InputLayer: convert batch_shape to input_shape
    if 'batch_shape' in config and 'input_shape' not in config:
        batch_shape = config['batch_shape']
        if batch_shape and len(batch_shape) > 1:
            config['input_shape'] = tuple(batch_shape[1:])
        config.pop('batch_shape', None)
    
    # Fix Embedding: convert DTypePolicy dict to string
    if 'dtype' in config and isinstance(config['dtype'], dict):
        dtype_config = config['dtype']
        if dtype_config.get('class_name') == 'DTypePolicy':
            inner_config = dtype_config.get('config', {})
            config['dtype'] = inner_config.get('name', 'float32')
        elif isinstance(dtype_config, dict) and 'module' in dtype_config:
            config['dtype'] = 'float32'  # Default fallback
    
    return config

class CompatibleInputLayer(BaseInputLayer):
    """InputLayer that accepts both batch_shape and input_shape for compatibility"""
    def __init__(self, input_shape=None, batch_shape=None, **kwargs):
        # Convert batch_shape to input_shape if provided
        if batch_shape is not None and input_shape is None:
            if len(batch_shape) > 1:
                input_shape = tuple(batch_shape[1:])
        # Remove batch_shape from kwargs to avoid passing it to parent
        kwargs.pop('batch_shape', None)
        super().__init__(input_shape=input_shape, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config = fix_layer_config(config)
        return super().from_config(config)

class CompatibleEmbedding(BaseEmbedding):
    """Embedding layer that handles old dtype policy format"""
    def __init__(self, dtype=None, **kwargs):
        # Handle old dtype format - convert DTypePolicy dict to string
        if isinstance(dtype, dict):
            dtype_config = dtype
            if dtype_config.get('class_name') == 'DTypePolicy':
                inner_config = dtype_config.get('config', {})
                dtype = inner_config.get('name', 'float32')
            elif 'module' in dtype_config:
                dtype = 'float32'  # Default fallback
        super().__init__(dtype=dtype, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config = fix_layer_config(config)
        return super().from_config(config)

# Patch the deserialization function to fix configs globally
try:
    from tensorflow.keras.engine.base_layer import Layer
    original_from_config = Layer.from_config
    
    @classmethod
    def patched_from_config(cls, config, custom_objects=None):
        # Fix the config before deserialization
        if isinstance(config, dict):
            config = fix_layer_config(config)
        return original_from_config(config, custom_objects)
    
    Layer.from_config = patched_from_config
except Exception as e:
    print(f"Could not patch Layer.from_config: {e}", file=sys.stderr)

# Custom objects for loading the model
custom_objects = {
    'InputLayer': CompatibleInputLayer,
    'Embedding': CompatibleEmbedding,
    'DTypePolicy': lambda **kwargs: 'float32',  # Dummy DTypePolicy handler
}

# Try loading the model with compatibility layers using custom_object_scope
try:
    with custom_object_scope(custom_objects):
        model = load_model(MODEL_FILENAME, compile=False)
except Exception as e:
    print(f"Model load with custom_object_scope failed: {e}", file=sys.stderr)
    # Fallback: try with custom_objects parameter
    try:
        model = load_model(MODEL_FILENAME, compile=False, custom_objects=custom_objects)
    except Exception as e2:
        print(f"Model load with custom_objects parameter failed: {e2}", file=sys.stderr)
        # Last resort: try using h5py to manually fix and reload
        try:
            import h5py
            # This is a complex workaround - for now, just raise with helpful message
            raise RuntimeError(
                f"Model loading failed due to version incompatibility. "
                f"Please re-save the model using the same TensorFlow version as deployment (2.11.0), "
                f"or provide the exact TF/Keras versions used during training so we can match them."
            )
        except Exception as e3:
            print(f"All model loading attempts failed: {e3}", file=sys.stderr)
            raise

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
