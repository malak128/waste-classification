import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from ultralytics import YOLO
import gdown
import glob
from tensorflow.keras.applications.resnet50 import preprocess_input



# Flask App

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# TrashNet Classes

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# Models Folder & Drive Link

MODEL_FOLDER = "saved_models"
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/17NT5-jTKiYdlFrP62o4Z2v1tjYMgjOJx?usp=sharing"
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Only download if folder is empty
if not os.listdir(MODEL_FOLDER):
    print("Downloading models from Google Drive...")
    gdown.download_folder(url=DRIVE_FOLDER_LINK, output=MODEL_FOLDER, quiet=False, use_cookies=False)
    print("✅ Models downloaded!")
else:
    print("Models already exist, skipping download.")

# Load TensorFlow/Keras Models

print("Loading TensorFlow/Keras models...")

autoencoder = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "denoising_autoencoder"))
custom_objects = {'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy}
multimodal_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "multimodal_resnet_text"),
                                              custom_objects=custom_objects)
cnn_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "cnn_model.h5"))
resnet_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "resnet_model_fixed.h5"))
generator_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "generator_model.h5"))
discriminator_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "discriminator_model.h5"))

print("TensorFlow/Keras models loaded!")

# Load YOLO Models Dynamically

chunk_folders = sorted(glob.glob(os.path.join(MODEL_FOLDER, "chunk", "chunk*")))
yolo_chunks = [YOLO(os.path.join(folder, "best.pt")) for folder in chunk_folders]
print(f"Loaded {len(yolo_chunks)} YOLO chunk models!")


# Helper Functions

def preprocess_image(img_path, model_name=None):
    """Preprocess image correctly for each model"""
    from PIL import Image
    import numpy as np

    if model_name in ["cnn_model", "autoencoder"]:
        target_size = (128, 128)
    else:  # For ResNet and multimodal
        target_size = (224, 224)

    img = Image.open(img_path).convert('RGB').resize(target_size)
    img_array = np.array(img).astype("float32")

    # Apply ResNet preprocessing for ResNet and multimodal
    if model_name in ["resnet_model", "multimodal_model"]:
        img_array = preprocess_input(img_array)
    else:
        img_array = img_array / 255.0  # scale to [0,1] for other models

    return np.expand_dims(img_array, axis=0)

def predict_yolo(image_path):
    all_labels = []
    for model in yolo_chunks:
        results = model.predict(source=image_path)
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                labels = [r.names[int(cls)] for cls in r.boxes.cls]
                all_labels.extend(labels)
    return all_labels

def generate_gan_image(label):
    label_index = class_names.index(label)
    latent_dim = 100
    noise = np.random.normal(0, 1, (1, latent_dim))
    label_array = np.array([[label_index]])
    generated_image = generator_model.predict([noise, label_array])
    gen_vis = ((generated_image[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8) if generated_image.min() < 0 else (generated_image[0] * 255).clip(0, 255).astype(np.uint8)
    save_name = f"gan_{label}.png"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    Image.fromarray(gen_vis).save(save_path)
    return save_name

def save_reconstruction(recon_array, original_filename="reconstructed.png"):
    recon = recon_array[0]
    vis = ((recon + 1.0) * 127.5).clip(0, 255).astype(np.uint8) if recon.min() < 0 else (recon * 255.0).clip(0, 255).astype(np.uint8)
    base, ext = os.path.splitext(original_filename)
    ext = ext or ".png"
    save_name = f"{base}_reconstructed{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    Image.fromarray(vis).save(save_path)
    return save_name

def get_prediction(file_path, model_name, text_input_value=None, gan_label=None):
    filename = None
    if model_name == "cnn_model":
        img = preprocess_image(file_path, model_name)
        preds = cnn_model.predict(img)
        prediction = class_names[int(np.argmax(preds, axis=1)[0])]
    elif model_name == "resnet_model":
        img = preprocess_image(file_path, model_name)
        preds = resnet_model.predict(img)
        prediction = class_names[int(np.argmax(preds, axis=1)[0])]
    elif model_name == "autoencoder":
        img = preprocess_image(file_path, model_name)
        reconstructed = autoencoder.predict(img)
        original_name = os.path.basename(file_path) if file_path else "reconstructed.png"
        filename = save_reconstruction(reconstructed, original_filename=original_name)
        prediction = "Autoencoder reconstructed image"
    elif model_name == "multimodal_model":
        img_array = preprocess_image(file_path, model_name)
        text_str = text_input_value or ""
        text_array = np.array([text_str], dtype=object)
        preds = multimodal_model.predict([img_array, text_array])
        prediction = class_names[int(np.argmax(preds, axis=1)[0])]
    elif model_name == "yolo":
        labels = predict_yolo(file_path)
        prediction = ", ".join(labels) if labels else "No objects detected"
    elif model_name == "gan":
        if gan_label in class_names:
            filename = generate_gan_image(gan_label)
            prediction = f"Generated GAN image for {gan_label}"
        else:
            prediction = "Invalid GAN label"
    else:
        prediction = "Unknown model"
    return prediction, filename


# Routes

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_file():
    selected_model = request.form.get('model')
    gan_label = request.form.get('gan_label')
    text_input_value = request.form.get('text_input')
    file_path = None
    filename = None
    if selected_model in ["cnn_model", "resnet_model", "autoencoder", "multimodal_model", "yolo"]:
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', prediction="⚠️ Please upload an image for this model.")
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    prediction, gen_filename = get_prediction(file_path, selected_model, text_input_value, gan_label)
    return render_template('index.html', prediction=prediction, filename=filename or gen_filename)


# Run App

if __name__ == '__main__':
    app.run(debug=True)

