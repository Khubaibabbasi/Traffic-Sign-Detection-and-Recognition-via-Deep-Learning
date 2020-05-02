# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:25:39 2019

@author: User
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

def read (mainFolder):
    xx = os.listdir(mainFolder)
    folder_list = []
    file_list = []
    
    for name in xx:
        if (len(name)==2):
            folder_list.append(name)
        elif ".ppm" in name :
            file_list.append(name)
            
    
    dataset = []
    for folder in folder_list:
        
        strTemp =  mainFolder + folder
        xx = os.listdir(strTemp)
        
#        if folder == "32" or folder =="15":
#            continue
        
        count = 0
        for name in xx:
            img = plt.imread(strTemp + "//" + name)
            dataset.append([strTemp + "//" + name,img])
            count = count + 1
            if count > len(file_list):# read 10 imges in each folder
                break
    
    
    annotation = {}
    with open(mainFolder+"gt.txt", "r") as ins:
        for line in ins:
            filename,x1,y1,x2,y2,l = line.split(';')
            
            if filename in annotation:
                annotation[filename].append([int(x1),int(y1),int(x2),int(y2)])
            else:
                annotation[filename] = [[int(x1),int(y1),int(x2),int(y2)]]
            
    return dataset, file_list, annotation
   
    
  
mainFolder = "FullIJCNN2013//"

dataset, file_list, annotation = read(mainFolder)

img = cv2.imread(mainFolder + file_list[8])

#plt.imshow(img)



