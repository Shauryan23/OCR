from flask import Flask, render_template, request
import numpy as np
import os
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from tensorflow import keras
import string

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="ocr_dr.tflite")

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# interpreter.allocate_tensors()

# # input details
# print(input_details)
# # output details
# print(output_details)

DEFAULT_ALPHABET = string.digits + string.ascii_lowercase
alphabets = DEFAULT_ALPHABET
blank_index = len(alphabets)

def run_tflite_model(image_path, quantization):
    input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_data = cv2.resize(image_path, (200, 31))
    input_data = input_data[np.newaxis]
    input_data = np.expand_dims(input_data, 3)
    input_data = input_data.astype('float32')/255
    path = f'ocr_{quantization}.tflite'
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print(file, "\t", filename)
        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)
        test_image = tf.keras.preprocessing.image.load_img(file_path)
        src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # print(src)

        # input_data = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        input_data = cv2.resize(src, (200, 31))
        input_data = input_data[np.newaxis]
        input_data = np.expand_dims(input_data, 3)
        input_data = input_data.astype('float32')/255
        path = 'ocr_dr.tflite'
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        final_output = "".join(alphabets[index] for index in output[0] if index not in [blank_index, -1])
        print(final_output)

        """ tflite_output = run_tflite_model(test_image, 'dr')
        final_output = "".join(alphabets[index] for index in tflite_output[0] if index not in [blank_index, -1])
        print(final_output) """

        # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # edged = cv2.Canny(blurred, 30, 150)
        # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # cnts = sort_contours(cnts, method="left-to-right")[0]
        # chars = []
        # for c in cnts:
        #     # compute the bounding box of the contour
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     # filter out bounding boxes, ensuring they are neither too small
        #     # nor too large
        #     if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        #         # extract the character and threshold it to make the character
        #         # appear as *white* (foreground) on a *black* background, then
        #         # grab the width and height of the thresholded image
        #         roi = gray[y:y + h, x:x + w]
        #         thresh = cv2.threshold(roi, 0, 255,
        #                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #         (tH, tW) = thresh.shape
        #         # if the width is greater than the height, resize along the
        #         # width dimension
        #         if tW > tH:
        #             thresh = imutils.resize(thresh, width=32)
        #         # otherwise, resize along the height
        #         else:
        #             thresh = imutils.resize(thresh, height=32)
        #         # re-grab the image dimensions (now that its been resized)
        #         # and then determine how much we need to pad the width and
        #         # height such that our image will be 32x32
        #         (tH, tW) = thresh.shape
        #         dX = int(max(0, 32 - tW) / 2.0)
        #         dY = int(max(0, 32 - tH) / 2.0)
        #         # pad the image and force 32x32 dimensions
        #         padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
        #                                     left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
        #                                     value=(0, 0, 0))
        #         padded = cv2.resize(padded, (32, 32))
        #         # prepare the padded image for classification via our
        #         # handwriting OCR model
        #         padded = padded.astype("float32") / 255.0
        #         padded = np.expand_dims(padded, axis=-1)
        #         # update our list of characters that will be OCR'd
        #         chars.append((padded, (x, y, w, h)))
        # boxes = [b[1] for b in chars]
        # chars = np.array([c[0] for c in chars], dtype="float32")
        # # OCR the characters using our handwriting recognition model
        # preds = model.predict(chars)
        # # define the list of label names
        # labelNames = "0123456789"
        # labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        # labelNames = [l for l in labelNames]
        # print(labelNames)
        # output = ""
        # for (pred, (x, y, w, h)) in zip(preds, boxes):
        #     i = np.argmax(pred)
        #     prob = pred[i]
        #     print(i, "\t", prob)
        #     label = labelNames[i]
        #     output += label

        # print("output",output)

        return render_template('sec.html', pred_output=final_output, user_image=file_path)


if __name__ == "__main__":
    app.run(threaded=False)