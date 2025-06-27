from flask import Flask, render_template, request
from werkzeug.utils import secure_filename  # Add this import
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load your trained model
model = tf.keras.models.load_model("model.h5")

# Disease classes
DISEASE_CLASSES = {
    0: {'name': 'Coccidiosis', 'info': 'A parasitic disease affecting the intestinal tract.'},
    1: {'name': 'Healthy', 'info': 'No signs of disease detected.'},
    2: {'name': 'Newcastle Disease', 'info': 'A highly contagious viral disease affecting birds.'},
    3: {'name': 'Salmonella', 'info': 'A bacterial infection causing digestive problems.'}
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'poultry_image' not in request.files:
            return render_template('predict.html', error="No file selected")
            
        file = request.files['poultry_image']
        if file.filename == '':
            return render_template('predict.html', error="No file selected")
            
        if file and allowed_file(file.filename):
            try:
                # Secure the filename and save
                filename = secure_filename(file.filename)
                upload_dir = app.config['UPLOAD_FOLDER']
                os.makedirs(upload_dir, exist_ok=True)
                img_path = os.path.join(upload_dir, filename)
                file.save(img_path)
                
                # Process the image
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                pred = model.predict(img_array)
                pred_class = np.argmax(pred, axis=1)[0]
                confidence = float(np.max(pred, axis=1)[0])
                
                # Get disease info
                disease = DISEASE_CLASSES.get(pred_class, {'name': 'Unknown', 'info': ''})
                
                return render_template('predict.html', 
                                    prediction=disease['name'],
                                    confidence=f"{confidence*100:.2f}%",
                                    info=disease['info'],
                                    image_path=img_path)
                
            except Exception as e:
                # Clean up the file if there was an error
                if 'img_path' in locals() and os.path.exists(img_path):
                    os.remove(img_path)
                return render_template('predict.html', error=f"Error processing image: {str(e)}")
        else:
            return render_template('predict.html', error="Invalid file type. Only JPG, PNG, JPEG allowed.")
    
    return render_template('predict.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)