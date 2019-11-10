import cv2
import matplotlib.pyplot as plt
# import numpy as np
import os


def extract_align_image(image, tooth_contour):
    """
    align each extracted tooth from the main image, to prevent tooth images
    from containing parts of the teeth in the vicinity.
    """
    rect = cv2.minAreaRect(tooth_contour)
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    m = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    rotated_image = cv2.warpAffine(src=image, M=m, dsize=(image.shape[1], image.shape[0]))
    cropped_image = cv2.getRectSubPix(image=rotated_image, patchSize=size, center=center)
    if abs(theta) > 45:
        cropped_image = cv2.rotate(src=cropped_image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    return cropped_image, theta


path = 'C://Users/Arman/Downloads/Compressed/dataset and code/train-val'
for mask in os.listdir(path=path+'/masks'):
    mask_name = mask.split('.')[0]
    for image in os.listdir(path=path+'/train2018'):
        image_name = image.split('.')[0]
        if image_name == mask_name:
            break
    this_image = cv2.imread(filename=path+'/train2018/'+image, flags=0)
    this_mask = cv2.imread(filename=path+'/masks/'+mask, flags=0)
    cnts, _ = cv2.findContours(image=this_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cnts))
    saving_path = "%s/teeth/%s" % (path, image_name)
    os.makedirs(saving_path, exist_ok=True)
    # print(saving_path)
    for idx, cnt in enumerate(cnts):
        if cv2.contourArea(contour=cnt) > 1000:
            this_tooth, angle = extract_align_image(image=this_image, tooth_contour=cnt)
            cv2.imwrite(filename='%s/%d.bmp' % (saving_path, idx), img=this_tooth)
            # print('%s/%d.bmp' % (saving_path, idx), angle)
