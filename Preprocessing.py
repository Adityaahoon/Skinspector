import numpy as np
import cv2
#import tensorflow as tf
import keras

labels=['Melanocytic nevi',
'Melanoma',
'Benign keratosis',
'Basal cell carcinoma',
'Actinic keratoses',
'Vascular lesions',
'Dermatofibroma']

def get_output(img):
    model=keras.models.load_model(r"D:\Vishal Kumar\Skinspector_layersproject\Data set")
    img=cv2.imread(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=np.resize(img,(28,28))
    img=img/255.0
    img=np.reshape(img,(-1,28,28,1))
    a=model.predict(img)
    ind=np.argmax(a)
    label=labels[ind]
    return label