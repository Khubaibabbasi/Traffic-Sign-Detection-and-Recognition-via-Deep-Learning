# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:58:46 2019

@author: Image processing part 
@ preprocessing data 
@ matching algorithem 
@ writing matching signs in external text file .0
"""
#Classical image processing technique to detect Objects or TS
#libraries
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import imutils
from imutils.object_detection import non_max_suppression
from timeit import default_timer as timer
from PIL import Image, ImageDraw
##################################### Model Library ###################################################
import tensorflow as tf
import keras
import glob


import read

def detect(img ,template,method):
    
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #0.65,1
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

    w,h= template.shape[::-1]

    res=cv2.matchTemplate(img_gray,template,method)

    threshold = 0.9 #0.45 0.36
    loc=np.where(res >= threshold)

    results=[]
    for pt in zip(*loc[::-1]):
        results.append([pt[0],pt[1], pt[0]+ w, pt[1]+ h])
        
    return results 

def showstore(count, cropped):# SHOW and cropped images
    #print(count)
    print(cropped.shape)
    cv2.imshow("Frame",cropped)
    width = 64
    height = 64
    dim = (width,height)
    
    resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
    #cv2.imwrite("Frame:" + str(count) + ".jpg",cropped)
    predict(resized)
    return

def predict(Croppedimage):   # predicting fun
    labels = ["Roadworks" , "Give way" , "Speed-50" , "Priority-Road", "Ahead Only", "Speed80", "Speed30", "Keep right"]
    image = cv2.resize(Croppedimage,(64,64))
    img=Croppedimage.reshape(-1,64,64,3)
    img = np.array(img).astype(np.float32)
    print(type(img))
    model = tf.keras.models.load_model("traffic_Sign.model")
    pred= model.predict_classes(img)
    print(labels[int(pred)])
    return
    
    
#for test_img in glob.glob("./Frames/*.jpg"):    
 #image = cv2.imread(test_img)
 #print(image)
 
     #image = cv2.imread("Croppedimage")
     #plt.imshow(image) 
    

mainFolder="D:/internship/Image Catching/FullIJCNN2013//" #dataset folder
dataset,file_list,annotation=read.read(mainFolder)
#print("length of dataset",len(dataset))
f=open("D:/internship/Image Catching/template-results.txt","+w")
# methods Choosen
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method=eval(methods[1])
count = 0
for i in range(0,len(file_list)):#len(file_list) read all imgs
    start=timer()
    img=cv2.imread(mainFolder + file_list[i])
    #cv2.imshow("frame",img)
    #cv2.waitKey(0)
    #break
    #cropimg=cv2.imread(mainFolder + file_list[i], cv2.IMREAD_UNCHANGED)
    #print(type(cropimg))
    #img1 = img
    #break
    #cropped = img1[773:815, 410:446]
    #plt.imshow("frame",cropped)
    #img=imutils.resize(img,width=min(800,img.shape[1]))

    results=[]
    for name,template in dataset:
#         print(name,template.shape)
            results_temp=detect(img.copy(),template,method)
            if (len(results_temp) >0):
                results.extend(results_temp)
                
      
       
    
    rects=np.array([[x1,y1,x2,y2] for (x1,y1,x2,y2) in results])
    #print(rects)
    pick= non_max_suppression(rects,probs=None,overlapThresh=0.2)
    #print(pick)
    #cv2.imshow("frame",img)
    #cv2.waitKey(0)
    #cropped = img[410:446,773:815]
    #cv2.imshow("frame1",cropped)
    #cv2.waitKey(0)
    #break
    '''
    for x1,y1,x2,y2 in pick:
        cropped=results_temp[x1:x2,y1:y2]
        showstore(count,cropped)
        count = count + 1
        break
    '''
    for x1,y1,x2,y2 in pick:
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
        #cv2.imshow("frame",img)
        #cv2.waitKey(0)
        f.write(file_list[i]+";"+str(x1)+";"+str(y1)+";"+str(x2)+";"+str(y2)+"\n")
        
        cropped=img[y1:y2,x1:x2]
        showstore(count,cropped)
        count = count + 1
        #break
        
        
     
    cv2.imshow("results",cv2.resize(img,(488,488)))
    #cv2.waitKey(0)
    #cv2.imshow("results",cropped)
    
    #img=imutils.resize(img,width=min(800,img.shape[1]))
    
    
    end=timer()
    print("elapsed time",end-start)
    print("results",pick)    
    k=cv2.waitKey(30) & 0xff
    if k ==27:
        break
#    if (len(results)>0):
##        cv2.waitKey(0)
        
f.close() #close file which is already opened
        
            
#    w,h,d=template.shape
#    img[0:w,0:h,:]=template
#        
#    print(len(pick))
#    plt.imshow(img)
#    cv2.waitKey(0)
#    




