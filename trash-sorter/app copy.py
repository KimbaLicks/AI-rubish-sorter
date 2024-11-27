from flask import Flask, render_template, request, jsonify
from llama_utils import llama32pi, encode_image
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Set up a folder for saving uploaded images (temporary location)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    image_file = request.files['image']
    if image_file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        image_file.save(image_path)

        # Convert image to base64
        base64_image = encode_image(image_path)

        prompt = ("Please identify any item of trash in this image that can be recycled."
              "For each item assign it a recycling material category "
              "(e.g., glass, metal, plastic, paper, etc.)")

        result = llama32pi(prompt, f"data:image/jpg;base64,{base64_image}")
        return jsonify(result=result)
    else:
        return jsonify(result="Failed to upload image.")


@app.route('/capture', methods=['POST'])
def capture_image():
    image_data = request.form['imageData']
    image_data = image_data.split(',')[1]  # remove the "data:image/jpg;base64," part

    # Save the captured image as a file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
    with open(image_path, 'wb') as f:
        f.write(base64.b64decode(image_data))

    # Process the image
    base64_image = encode_image(image_path)

    prompt = ("Please identify any item of trash in this image that can be recycled."
              "For each item assign it a recycling material category "
              "(e.g., glass, metal, plastic, paper, etc.)")
    

    result = llama32pi(prompt, f"data:image/jpg;base64,{base64_image}")
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
