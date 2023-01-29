from tensorflow.keras.models import load_model
import tensorflow
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-loc', required=True, help = 'Image file location ')
args = ap.parse_args()

classifier = load_model('model.h5')
img = cv2.resize(cv2.imread(args.loc), (64,64))
image = cv2.resize(cv2.imread(args.loc), (128, 128))
img = img/255

img = np.array([img])
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.

font = cv2.FONT_HERSHEY_SIMPLEX

if(prediction[:,:]>0.5):
    value ='Dog :%1.2f'%(prediction[0,0])
    # Using cv2.putText() method
    image = cv2.rectangle(image, (0, 100), (128, 128), (255, 255, 255), -1)
    image = cv2.putText(image, value, (47, 110), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
    # plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

else:
    value ='Cat :%1.2f'%(1.0-prediction[0,0])
    image = cv2.rectangle(image, (0, 100), (128, 128), (255, 255, 255), -1)
    image = cv2.putText(image, value, (47, 110), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
    # plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

print(value)
cv2.imshow('Output', image)
cv2.waitKey(0)