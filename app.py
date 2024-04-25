from flask import Flask, render_template, request
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model_path = './models/Densenet_comb.h5'
def build_model():
    # Load the DenseNet121 base model without top layers
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Add custom top layers for classification
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)  # Assuming 5 output classes
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Build the model
model = build_model()
model.load_weights(model_path)


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    result = None  # Initialize result variable
    filename = None  # Initialize filename variable
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {file_path}")

            # Save the file
            file.save(file_path)

            # Check if the file was saved
            if os.path.isfile(file_path):
                print("File saved successfully!")
            else:
                print("Error: File not saved.")

            prediction = predict_image(file_path, model)
            result = "Prediction: " + prediction

    return render_template('index.html', result=result, filename=filename)

def predict_image(img_path, model):
    label_title = {"0" : "No DR","1" : "Mild","2" : "Moderate","3" :"Severe","4" : "Proliferative DR"}
    class_labels=["No DR","Mild","Moderate","Severe","Proliferative DR"]
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    class_label = np.argmax(prediction)
    return class_labels[class_label]

if __name__ == '__main__':
    app.run(debug=True)
