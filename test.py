import glob
import os
import cv2.cv2 as cv2
import numpy as np

# path = './images/'
# for idx, file in enumerate(glob.glob(pathname=path+'**.jpg')):
#     os.rename(src=file, dst=path+'%d.jpg' % idx)
#
# path = './extracted_teeth/'
# for idx, folder in enumerate(os.listdir(path=path)):
#     os.rename(src=path + folder, dst=path + '%d' % idx)
#
# path = './extracted_teeth/'
# for folder in os.listdir(path=path):
#     subpath = path + folder
#     for file_idx, file in enumerate(glob.glob(pathname=subpath+'/**.bmp')):
#         os.rename(src=file, dst=subpath+'/%d-%d.bmp' % (int(folder), file_idx))

path = './temp/'
for file in glob.glob(pathname=path+'**.jpg'):
    img = cv2.imread(filename=file, flags=0)
    h, w = img.shape[:2]
    if h <= 300:
        h_lower, h_upper = int((h/2.0) - (h/3)), int((h/2.0) + (h/3))
        w_lower, w_upper = int((w / 2.0) - (w / 4)), int((w / 2.0) + (w / 4))
    else:
        h_lower, h_upper = int((h/2.0) - (h/2.5)), int((h/2.0) + (h/2.5))
        w_lower, w_upper = int((w / 2.0) - (w / 3.25)), int((w / 2.0) + (w / 3.25))
    img_crp = img[h_lower:h_upper, w_lower:w_upper]
    img_resz = cv2.resize(src=img_crp, dsize=(400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename='./temp_new/%s' % file[file.rfind("\\")+1:], img=img_resz)

