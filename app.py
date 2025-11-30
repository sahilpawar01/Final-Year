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
    def from_config(cls, config, custom_objects=None):
        # Accept custom_objects parameter but don't pass it to parent
        # TF 2.11's from_config only accepts (cls, config)
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
    def from_config(cls, config, custom_objects=None):
        # Accept custom_objects parameter but don't pass it to parent
        # TF 2.11's from_config only accepts (cls, config)
        config = fix_layer_config(config)
        return super().from_config(config)

# Patch the deserialization function to fix configs globally
# Try different module paths for different TF versions and patch all of them
def patch_layer_from_config():
    """Patch Layer.from_config in all possible locations"""
    layer_modules = [
        'keras.engine.base_layer',
        'tensorflow.python.keras.engine.base_layer',
        'tf_keras.engine.base_layer',
        'keras.src.engine.base_layer',
        'tensorflow.keras.engine.base_layer',
    ]
    
    for module_path in layer_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, 'Layer'):
                Layer = module.Layer
                if hasattr(Layer, 'from_config'):
                    # Get the original method - handle both bound and unbound
                    original_from_config = Layer.from_config
                    # Get the underlying function if it's a classmethod descriptor
                    if hasattr(original_from_config, '__func__'):
                        original_func = original_from_config.__func__
                    else:
                        original_func = original_from_config
                    
                    @classmethod
                    def patched_from_config(cls, *args, **kwargs):
                        # Handle any call signature - extract config from args or kwargs
                        # Keras may call as: from_config(config) or from_config(config, custom_objects=...)
                        if len(args) > 0:
                            config = args[0]
                        elif 'config' in kwargs:
                            config = kwargs['config']
                        else:
                            # Fallback - try to get from first positional arg
                            config = args[0] if args else {}
                        
                        # Fix the config before deserialization
                        if isinstance(config, dict):
                            config = fix_layer_config(config)
                        
                        # In TF 2.11, from_config only accepts (cls, config)
                        # custom_objects is handled at a higher level in load_model
                        # Call the original function with only cls and config
                        try:
                            return original_func(cls, config)
                        except TypeError as te:
                            # If that fails, try without cls (some versions might bind it differently)
                            print(f"Warning: from_config call failed with cls, trying without: {te}", file=sys.stderr)
                            return original_func(config)
                    
                    Layer.from_config = patched_from_config
                    print(f"Successfully patched Layer.from_config in {module_path}", file=sys.stderr)
        except (ImportError, AttributeError) as e:
            continue
        except Exception as e:
            print(f"Error patching {module_path}: {e}", file=sys.stderr)
            continue

# Apply the patch BEFORE any model loading
patch_layer_from_config()

# Also patch deserialization functions that might call from_config
def patch_deserialization_functions():
    """Patch deserialization functions to handle config fixes"""
    deserialization_modules = [
        'tensorflow.python.keras.saving.model_config',
        'keras.src.saving.saving_lib',
        'tensorflow.keras.utils',
    ]
    
    for module_path in deserialization_modules:
        try:
            module = importlib.import_module(module_path)
            # Patch model_from_config if it exists
            if hasattr(module, 'model_from_config'):
                original_model_from_config = module.model_from_config
                def patched_model_from_config(config, custom_objects=None):
                    # Fix any layer configs in the model config
                    if isinstance(config, dict) and 'config' in config:
                        model_config = config.get('config', {})
                        if 'layers' in model_config:
                            for layer_config in model_config['layers']:
                                if isinstance(layer_config, dict) and 'config' in layer_config:
                                    layer_config['config'] = fix_layer_config(layer_config['config'])
                    return original_model_from_config(config, custom_objects)
                module.model_from_config = patched_model_from_config
                print(f"Patched model_from_config in {module_path}", file=sys.stderr)
        except (ImportError, AttributeError):
            continue
        except Exception as e:
            print(f"Error patching {module_path}: {e}", file=sys.stderr)
            continue

patch_deserialization_functions()

# Patch functional model reconstruction to handle string input_data
def patch_functional_reconstruction():
    """Patch functional model reconstruction to handle string input_data in node_data"""
    try:
        from tensorflow.python.keras.engine import functional as tf_functional
        import tensorflow.python.keras.utils.nest as nest
        
        # Create a compatibility wrapper for ListWrapper that handles strings
        class CompatibleListWrapper:
            """Wrapper that can handle both ListWrapper objects and strings"""
            def __init__(self, data):
                self._data = data
            
            def as_list(self):
                if isinstance(self._data, str):
                    # String format: assume it's a layer name, return [name, 0, 0]
                    return [self._data, 0, 0]
                elif hasattr(self._data, 'as_list'):
                    # It's a real ListWrapper, call its as_list method
                    return self._data.as_list()
                elif isinstance(self._data, (list, tuple)):
                    # Already a list
                    return list(self._data)
                else:
                    # Fallback: wrap in list
                    return [self._data, 0, 0]
        
        # Get the original reconstruct_from_config
        original_reconstruct = tf_functional.reconstruct_from_config
        
        # Store original process_node if it exists in the closure
        def patched_reconstruct_from_config(config, custom_objects=None, created_layers=None):
            """Patched version that wraps process_node to handle strings"""
            # We need to patch the inner process_node function
            # This is tricky because it's defined inside reconstruct_from_config
            # So we'll patch it by wrapping the entire reconstruction process
            
            # First, let's try to fix the config if it has string-based node data
            if isinstance(config, dict) and 'layers' in config:
                for layer_data in config['layers']:
                    if isinstance(layer_data, dict) and 'inbound_nodes' in layer_data:
                        inbound_nodes = layer_data['inbound_nodes']
                        # The inbound_nodes should already be in the right format
                        # The issue is during processing, not in the config
            
            # Patch ListWrapper class to handle strings
            try:
                from tensorflow.python.keras.utils.generic_utils import ListWrapper
                original_list_wrapper_init = ListWrapper.__init__
                original_list_wrapper_as_list = ListWrapper.as_list
                
                def patched_list_wrapper_init(self, seq=None):
                    # If seq is a string, convert it to a list
                    if isinstance(seq, str):
                        seq = [seq, 0, 0]
                    original_list_wrapper_init(self, seq)
                
                def patched_list_wrapper_as_list(self):
                    # If the wrapped data is a string, return it as a list
                    if hasattr(self, '_list') and isinstance(self._list, str):
                        return [self._list, 0, 0]
                    return original_list_wrapper_as_list(self)
                
                ListWrapper.__init__ = patched_list_wrapper_init
                ListWrapper.as_list = patched_list_wrapper_as_list
                print("Patched ListWrapper to handle strings", file=sys.stderr)
            except (ImportError, AttributeError) as e:
                print(f"Could not patch ListWrapper: {e}", file=sys.stderr)
            
            # Also patch nest.flatten to wrap strings
            original_flatten = nest.flatten
            
            def patched_flatten(structure):
                result = original_flatten(structure)
                # Wrap any strings in CompatibleListWrapper
                wrapped_result = []
                for item in result:
                    if isinstance(item, str):
                        wrapped_result.append(CompatibleListWrapper(item))
                    else:
                        wrapped_result.append(item)
                return wrapped_result
            
            # Temporarily replace nest.flatten
            nest.flatten = patched_flatten
            
            try:
                result = original_reconstruct(config, custom_objects, created_layers)
            finally:
                # Restore original
                nest.flatten = original_flatten
            
            return result
        
        tf_functional.reconstruct_from_config = patched_reconstruct_from_config
        print("Patched functional.reconstruct_from_config to handle string input_data", file=sys.stderr)
        
    except Exception as e:
        print(f"Could not patch functional reconstruction: {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)

patch_functional_reconstruction()

# Create a proper DTypePolicy class for deserialization
# Try to use the real DTypePolicy if available, otherwise create a compatibility class
try:
    from tensorflow.keras.mixed_precision import Policy as DTypePolicyCompat
    print("Using TensorFlow DTypePolicy (Policy)", file=sys.stderr)
except ImportError:
    try:
        from keras.dtype_policies import DTypePolicy as DTypePolicyCompat
        print("Using Keras DTypePolicy", file=sys.stderr)
    except ImportError:
        # Fallback: Create a compatibility class with all required attributes
        class DTypePolicyCompat:
            """Compatibility class for DTypePolicy deserialization"""
            def __init__(self, name='float32', **kwargs):
                if isinstance(name, dict):
                    # Handle config dict directly
                    name = name.get('name', 'float32') if isinstance(name, dict) else 'float32'
                
                self._name = str(name) if name else 'float32'
                # Parse name to get compute and variable dtypes
                if self._name == "mixed_float16":
                    self._compute_dtype = "float16"
                    self._variable_dtype = "float32"
                elif self._name == "mixed_bfloat16":
                    self._compute_dtype = "bfloat16"
                    self._variable_dtype = "float32"
                else:
                    # Default to the same dtype for both
                    self._compute_dtype = self._name
                    self._variable_dtype = self._name
                
                # Also set as direct attributes for compatibility
                self.compute_dtype = self._compute_dtype
                self.variable_dtype = self._variable_dtype
            
            @property
            def name(self):
                return self._name
            
            @classmethod
            def from_config(cls, config):
                if isinstance(config, dict):
                    name = config.get('name', 'float32')
                else:
                    name = str(config) if config else 'float32'
                return cls(name=name)
            
            def __repr__(self):
                return f"<DTypePolicyCompat '{self._name}'>"
        
        print("Using custom DTypePolicyCompat class", file=sys.stderr)

# Custom objects for loading the model
custom_objects = {
    'InputLayer': CompatibleInputLayer,
    'Embedding': CompatibleEmbedding,
    'DTypePolicy': DTypePolicyCompat,
}

# Try loading the model with compatibility layers
print("Starting model loading process...", file=sys.stderr)
try:
    print("Attempting to load model with custom_object_scope...", file=sys.stderr)
    with custom_object_scope(custom_objects):
        model = load_model(MODEL_FILENAME, compile=False)
    print("✓ Model loaded successfully with custom_object_scope!", file=sys.stderr)
except Exception as e:
    print(f"✗ Model load with custom_object_scope failed: {e}", file=sys.stderr)
    print(f"  Error type: {type(e).__name__}", file=sys.stderr)
    import traceback
    print(f"  Traceback: {traceback.format_exc()}", file=sys.stderr)
    
    # Fallback: try with custom_objects parameter
    try:
        print("Attempting to load model with custom_objects parameter...", file=sys.stderr)
        model = load_model(MODEL_FILENAME, compile=False, custom_objects=custom_objects)
        print("✓ Model loaded successfully with custom_objects parameter!", file=sys.stderr)
    except Exception as e2:
        print(f"✗ Model load with custom_objects parameter failed: {e2}", file=sys.stderr)
        print(f"  Error type: {type(e2).__name__}", file=sys.stderr)
        import traceback
        print(f"  Traceback: {traceback.format_exc()}", file=sys.stderr)
        
        # Final error with helpful message
        raise RuntimeError(
            f"Model loading failed due to version incompatibility.\n"
            f"Error 1: {str(e)}\n"
            f"Error 2: {str(e2)}\n\n"
            f"Please re-save the model using TensorFlow 2.11.0, "
            f"or provide the exact TF/Keras versions used during training so we can match them."
        )

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
