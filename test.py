import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'C://Users/Arman/Downloads/Compressed/dataset and code/train-val'
for mask in os.listdir(path=path+'/masks'):
    this_mask = cv2.imread(filename=path+'/masks/'+mask, flags=0)
    cnts, _ = cv2.findContours(image=this_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in cnts:
        if cv2.contourArea(contour=cnt) > 30000:
            print(cv2.contourArea(contour=cnt))
            i += 1
    if i > 0:
        print(mask)

