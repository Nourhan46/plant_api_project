import os
import numpy as np
from PIL import Image
import tensorflow as tf  # Used for model conversion (in a separate script)
from flask import Flask, request, jsonify
import logging
from io import BytesIO

# --- Configuration ---
MODEL_DIR = 'tflite_models'
MODEL1_PATH = os.path.join(MODEL_DIR, 'model1.tflite')  # Updated to .tflite
MODEL2_PATH = os.path.join(MODEL_DIR, 'model2.tflite')  # Updated to .tflite
MODEL3_PATH = os.path.join(MODEL_DIR, 'model3.tflite')  # Updated to .tflite

IMG_SIZE_MODEL1_3 = (300, 300)
IMG_SIZE_MODEL2 = (200, 200)

MODEL1_CLASSES = ['class1', 'class2', 'class3']
MODEL2_CLASSES = [
    'Apple___Apple_scab-unhealthy',
    'Apple___Black_rot-unhealthy',
    'Apple___Cedar_apple_rust-unhealthy',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew-unhealthy',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot-unhealthy',
    'Corn_(maize)___Common_rust_-unhealthy',
    'Corn_(maize)___Northern_Leaf_Blight--unhealthy',
    'Corn_(maize)___healthy',
    'Grape___Black_rot-unhealthy',
    'Grape___Esca_(Black_Measles)--unhealthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)-unhealthy',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)-unhealthy',
    'Peach___Bacterial_spot--unhealthy',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot -unhealthy',
    'Pepper,_bell___healthy',
    'Potato___Early_blight -unhealthy',
    'Potato___Late_blight -unhealthy',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew -unhealthy',
    'Strawberry___Leaf_scorch -unhealthy',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot -unhealthy',
    'Tomato___Early_blight -unhealthy',
    'Tomato___Late_blight -unhealthy',
    'Tomato___Leaf_Mold -unhealthy',
    'Tomato___Septoria_leaf_spot -unhealthy',
    'Tomato___Spider_mites Two-spotted_spider_mite -unhealthy',
    'Tomato___Target_Spot -unhealthy',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus -unhealthy',
    'Tomato___Tomato_mosaic_virus -unhealthy',
    'Tomato___healthy'
]
MODEL3_CLASSES = [
    'castor_oil_plant', 'dieffenbachia', 'foxglove', 'lilies',
    'lily_of_the_valley', 'oleander', 'rhubarb', 'wisteria'
]

# --- Initialize Flask App ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Load Models ---
# Load TFLite models
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        app.logger.error(f"Error loading TFLite model {model_path}: {e}")
        return None

try:
    app.logger.info("Loading TFLite models...")
    model1_interpreter = load_tflite_model(MODEL1_PATH)
    model2_interpreter = load_tflite_model(MODEL2_PATH)
    model3_interpreter = load_tflite_model(MODEL3_PATH)

    if not all([model1_interpreter, model2_interpreter, model3_interpreter]):
        app.logger.error("Failed to load one or more TFLite models.")
        exit()  # Exit if any model fails to load

    app.logger.info("TFLite models loaded successfully.")
except Exception as e:
    app.logger.error(f"Fatal error loading models: {e}")
    exit()

# --- Helper Function: Preprocess Image ---
def preprocess_image(image, target_size):
    """
    Resizes, converts to numpy array, rescales, and adds batch dimension.

    Args:
        image (PIL.Image.Image): The image to preprocess.
        target_size (tuple): The desired size (height, width).

    Returns:
        numpy.ndarray: The preprocessed image array, or None on error.
    """
    try:
        img_copy = image.copy()
        if img_copy.mode != "RGB":
            img_copy = img_copy.convert("RGB")
        img_copy = img_copy.resize(target_size)
        image_array = np.array(img_copy, dtype=np.float32)  # Ensure float32 for TFLite
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        app.logger.error(f"Error preprocessing image to size {target_size}: {e}")
        return None

# --- Helper Function: Run Inference ---
def run_inference(interpreter, input_data):
    """
    Runs inference with a TFLite interpreter.

    Args:
        interpreter (tf.lite.Interpreter): The TFLite interpreter.
        input_data (numpy.ndarray): The input data for the model.

    Returns:
        numpy.ndarray: The output of the model, or None on error.
    """
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Resize input tensor if necessary
        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    except Exception as e:
        app.logger.error(f"Error running inference: {e}")
        return None

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for plant disease/toxicity prediction using TensorFlow Lite.
    """
    if 'file' not in request.files:
        app.logger.warning("Prediction request failed: No file part.")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        app.logger.warning("Prediction request failed: No selected file.")
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_data = file.read()
        original_image = Image.open(BytesIO(image_data))
        app.logger.info(f"Received image file: {file.filename}")

    except Exception as e:
        app.logger.error(f"Error opening image file {file.filename}: {e}")
        return jsonify({'error': 'Could not read image file.'}), 400

    # --- Stage 1: Predict with Model 1 ---
    processed_img_m1 = preprocess_image(original_image, IMG_SIZE_MODEL1_3)
    if processed_img_m1 is None:
        return jsonify({'error': 'Failed to preprocess image for Model 1.'}), 500

    try:
        app.logger.info("Predicting with Model 1...")
        prediction_m1 = run_inference(model1_interpreter, processed_img_m1)
        if prediction_m1 is None or prediction_m1.size == 0 or prediction_m1.shape[1] != len(MODEL1_CLASSES):
            app.logger.error(
                f"Model 1 prediction output invalid. Expected {len(MODEL1_CLASSES)} classes, got shape {prediction_m1.shape if prediction_m1 is not None else 'None'}"
            )
            return jsonify({'error': 'Internal server error: Model 1 prediction failed.'}), 500

        predicted_class_index_m1 = np.argmax(prediction_m1[0])
        if predicted_class_index_m1 >= len(MODEL1_CLASSES):
            app.logger.error(f"Model 1 predicted index out of bounds: {predicted_class_index_m1}")
            return jsonify({'error': 'Internal server error: Model 1 index error.'}), 500

        predicted_class_name_m1 = MODEL1_CLASSES[predicted_class_index_m1]
        app.logger.info(f"Model 1 prediction: {predicted_class_name_m1}")

        final_result = "Classification could not be completed."
        status_code = 500

        # --- Stage 2: Conditional Prediction ---
        if predicted_class_name_m1 == 'class1':
            app.logger.info("Using Model 2...")
            processed_img_m2 = preprocess_image(original_image, IMG_SIZE_MODEL2)
            if processed_img_m2 is None:
                return jsonify({'error': 'Failed to preprocess image for Model 2.'}), 500

            prediction_m2 = run_inference(model2_interpreter, processed_img_m2)
            if prediction_m2 is None or prediction_m2.size == 0 or prediction_m2.shape[1] != len(MODEL2_CLASSES):
                app.logger.error(
                    f"Model 2 prediction output invalid. Expected {len(MODEL2_CLASSES)} classes, got shape {prediction_m2.shape if prediction_m2 is not None else 'None'}"
                )
                return jsonify({'error': 'Internal server error: Model 2 prediction failed.'}), 500

            predicted_class_index_m2 = np.argmax(prediction_m2[0])
            if predicted_class_index_m2 >= len(MODEL2_CLASSES):
                app.logger.error(f"Model 2 predicted index out of bounds: {predicted_class_index_m2}")
                return jsonify({'error': 'Internal server error: Model 2 index error.'}), 500

            plant_info = MODEL2_CLASSES[predicted_class_index_m2]
            final_result = f"{plant_info}"
            status_code = 200

        elif predicted_class_name_m1 == 'class2':
            app.logger.info("Using Model 3...")
            processed_img_m3 = preprocess_image(original_image, IMG_SIZE_MODEL1_3)
            if processed_img_m3 is None:
                return jsonify({'error': 'Failed to preprocess image for Model 3.'}), 500

            prediction_m3 = run_inference(model3_interpreter, processed_img_m3)
            if prediction_m3 is None or prediction_m3.size == 0 or prediction_m3.shape[1] != len(MODEL3_CLASSES):
                app.logger.error(
                    f"Model 3 prediction output invalid. Expected {len(MODEL3_CLASSES)} classes, got shape {prediction_m3.shape if prediction_m3 is not None else 'None'}"
                )
                return jsonify({'error': 'Internal server error: Model 3 prediction failed.'}), 500

            predicted_class_index_m3 = np.argmax(prediction_m3[0])
            if predicted_class_index_m3 >= len(MODEL3_CLASSES):
                app.logger.error(f"Model 3 predicted index out of bounds: {predicted_class_index_m3}")
                return jsonify({'error': 'Internal server error: Model 3 index error.'}), 500

            plant_name = MODEL3_CLASSES[predicted_class_index_m3]
            final_result = f"{plant_name} - toxic"
            status_code = 200

        elif predicted_class_name_m1 == 'class3':
            app.logger.info("Model 1 predicted class3 - returning 'Unknown'.")
            final_result = "Unknown, please enter a valid picture."
            status_code = 200

        else:
            app.logger.error(f"Unexpected prediction from Model 1: {predicted_class_name_m1}")
            final_result = "Internal server error: Unexpected intermediate result."
            status_code = 500

        app.logger.info(f"Final result: '{final_result}', Status: {status_code}")
        return jsonify({'prediction': final_result}), status_code

    except Exception as e:
        app.logger.error(f"Unhandled exception during prediction: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

