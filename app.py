from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import gdown

if not os.path.exists("animal_model.h5"):
    url = "https://drive.google.com/file/d/1q1fSPQhsCZP2FtSuR3CndKVhIPY6oYQn/view?usp=drive_link"
    gdown.download(url, "animal_model.h5", quiet=False)
# Load mô hình
model = tf.keras.models.load_model('animal_model.h5')

# Kích thước ảnh đầu vào
img_height, img_width = 128, 128

# Lấy danh sách class
class_names = ['cat', 'dog', 'monkey', 'lion', 'tiger', 'elephant', 'deer', 'zebra']

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Trang chính
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Dự đoán
            img = Image.open(filepath).convert('RGB')
            img = img.resize((img_width, img_height))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Tạo batch
            img_array = img_array / 255.0

            predictions = model.predict(img_array)[0]
            top_3_indices = np.argsort(predictions)[::-1][:3]
            top_3 = [(class_names[i], round(predictions[i]*100, 2)) for i in top_3_indices]

            return render_template('index.html', filename=filename, result=top_3)

    return render_template('index.html')

# Route hiển thị ảnh
@app.route('/display/<filename>')
def display_image(filename):
    return f'/static/uploads/{filename}'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
