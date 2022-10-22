import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image
from numpy import float32
(x_train,y_train),(x_test,y_test)=mnist.load_data()


print(x_train.shape)
x_trainr=x_train.reshape(-1,28*28).astype(float32)/255.0
x_testr=x_test.reshape(-1,28*28).astype(float32)/255.0

#sequential API convinient not so flexible
model=keras.Sequential([
                        layers.Dense(512,activation='relu'),
                        layers.Dense(256,activation='relu'),                     
                        layers.Dense(10)
])
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'] )
#tensorboard_cb= tf.keras.callbacks.TensorBoard(run_logdir)

history=model.fit(x_trainr,y_train,verbose=2,batch_size=52,epochs=10)
model.evaluate(x_testr,y_test,batch_size=52,verbose=2)

import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
print(plt.show())
model.save('mnist_num.h5')