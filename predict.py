import cv2
import numpy as np
import tensorflow as tf
#import keras

model = tf.keras.models.load_model("traffic_Sign.model")
labels = ["Roadworks" , "Give way" , "Speed-50" , "Priority-Road", "Ahead Only", "Speed80", "Speed30", "Keep right"]


image = cv2.imread("roadwork.jpg") 
image=cv2.resize(image,(256,256))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)
#print("****************************************")

image = cv2.imread("Giveway.jpg") 
image=cv2.resize(image,(256,256))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)
#print("****************************************")


image = cv2.imread("506.png") 
image=cv2.resize(image,(512,512))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)
#print("****************************************")

image = cv2.imread("roadpriority.jpg") 
image=cv2.resize(image,(256,256))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)
#print("****************************************")
image = cv2.imread("a1.png") 
image=cv2.resize(image,(256,256))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)
#print("****************************************")

image = cv2.imread("801.jpg") 
image=cv2.resize(image,(256,256))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)
#print("****************************************")


image = cv2.imread("301.png") 
image=cv2.resize(image,(512,512))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)
#print("****************************************")

image = cv2.imread("k4.jpg") 
image=cv2.resize(image,(512,512))
image1 = cv2.resize(image,(64,64))
image1= image1/255
img=image1.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict_classes(img)
print(labels[int(pred)])
cv2.imshow('IMAGES',image)
cv2.waitKey(0)