from flask import Flask, jsonify, request, render_template
import re
import base64
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import io
import sys

sys.path.append('CNN_model/')
import CNN_model

app = Flask(__name__)

#removes the transparency/Alpha channel by copying the RGB channels to a new image
def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency 
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL 
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def grab_image():
	image_base64 = request.values['imageBase64']
	image_data = base64.decodebytes(re.sub('^data:image/.+;base64,', '', image_base64).encode())
	image = Image.open(io.BytesIO(image_data))

	# resize to 28x28, remove transparency and convert to grayscale
	#image = image.resize((28,28))
	image = remove_transparency(image).convert('L')

	image = np.asarray(image)
	#mnist dataset contains white numbers with dark backgrounds, so an inversion is necessary
	image = np.invert(image)
	image = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
	image = image.reshape(28,28,1)
	image = np.expand_dims(image, axis=0) #(1,28,28,1) to match model's input
	image = image.astype('float32')
	prediction = CNN_model.predict(image)
	CNN_model.clear_session()
	# im = Image.fromarray(image)
	# im.save("image.png")
	return str(prediction)

if __name__ == '__main__':
	#Threaded false is a workaround to a currently open bug in the Keras Framework that causes issues with Tensorflow and Flask
	# app.run(host="0.0.0.0",port=5000)
	app.run(debug=True, use_reloader=False)
