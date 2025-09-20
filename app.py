from flask import Flask, render_template, request, send_from_directory, flash, url_for
from pathlib import Path
import os
import pandas as pd
import base64
from io import BytesIO
from PIL import Image

from src.duckeye import DuckEYE
from src.config import DEVICE, INPUT_PATH, OUTPUT_PATH
import src.preprocessing.preprocessing as prep

app = Flask(__name__)
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = INPUT_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure directories exist
INPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)

duckeye = DuckEYE(device=DEVICE)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = file.filename
        filepath = INPUT_PATH / filename
        file.save(filepath)

    # Check if a camera-captured image was sent
    elif 'image' in request.form:
        image_data = request.form['image']
        # Remove the data URI prefix (e.g., "data:image/jpeg;base64,")
        image_data = image_data.split(',')[1]
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        # Generate a unique filename for the captured image
        filename = f"capture_{len(list(INPUT_PATH.glob('capture_*')))}.jpg"
        filepath = INPUT_PATH / filename
        # Save the image using PIL
        image = Image.open(BytesIO(image_bytes))
        image.save(filepath, 'JPEG')

    else:
        flash('No file or image provided')
        return render_template('upload.html')

    # Process the image (file or captured)
    prep.resize_single(filename)
    results = duckeye.observe_single(filename)
    df = pd.DataFrame([results])
    csv_path = OUTPUT_PATH / f"{filename}_results.csv"
    df.to_csv(csv_path, index=False)

    return render_template('results.html', results=results, csv_url=url_for('download_csv', filename=filename))

@app.route('/download/<filename>')
def download_csv(filename):
    return send_from_directory(OUTPUT_PATH, f"{filename}_results.csv")

if __name__ == '__main__':
    app.run(debug=False)