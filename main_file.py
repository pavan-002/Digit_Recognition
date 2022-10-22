from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
from numpy import float32
import tensorflow as tf
from PIL import Image 
import gradio as gr
import os

new_model =tf.keras.models.load_model('mnist_num.h5')
from tensorflow.keras.datasets import mnist
(x,y),(a,b)=mnist.load_data()
ar=a.reshape(-1,28*28).astype(float32)/255.0
#x_train,y_train ,x_test,y_test

import tensorflow as tf
import numpy as np
from urllib.request import urlretrieve
import gradio as gr

#urlretrieve("https://gr-models.s3-us-west-2.amazonaws.com/mnist-model.h5", "mnist-model.h5")
model = tf.keras.models.load_model("mnist_num.h5")

def recognize_digit(image):
    image = image.reshape(1, -1)  # add a batch dimension
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

output_component = gr.outputs.Label(num_top_classes=3)

gr.Interface(fn=recognize_digit, 
             inputs="sketchpad", 
             outputs=output_component,
             title="MNIST Sketchpad",
             description="Draw a number 0 through 9 on the sketchpad, and click submit to see the model's predictions\n\ncreated by:Pannaga &team.",
             ).launch(share=True)
