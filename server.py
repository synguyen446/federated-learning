from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from utils import get_most_recent_file_path


app = Flask(__name__)
CORS(app)

# Sample route
@app.route("/")
def home():
    return {"message":"Hello World"}

@app.route('/predict', methods=['GET'])
def predict():
    image_path = get_most_recent_file_path()
    image = load_img(rf"{image_path}",target_size = (28,28), color_mode ="grayscale")
    image_array = img_to_array(image)
    image_array /= 255

    image= np.array(image_array).reshape(-1,28,28,1)
    

    model = tf.keras.models.load_model(r"C:\Users\syngu\GitHub\sign-language-translator\flask-server-api\flask-server\model\centralized-model.keras")

    letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
    }

    y_predict = model.predict(image)
    max_val = (0,0)
    for i in range(len(y_predict[0])):
        if y_predict[0][i] > max_val[0]:
            max_val = (y_predict[0][i],i)

    
    return jsonify(letter[max_val[1]])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
