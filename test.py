import glob
import os

path = './images/'
for idx, file in enumerate(glob.glob(pathname=path+'**.jpg')):
    os.rename(src=file, dst=path+'%d.jpg' % idx)

path = './extracted_teeth/'
for idx, folder in enumerate(os.listdir(path=path)):
    os.rename(src=path + folder, dst=path + '%d' % idx)

path = './extracted_teeth/'
for folder in os.listdir(path=path):
    subpath = path + folder
    for file_idx, file in enumerate(glob.glob(pathname=subpath+'/**.bmp')):
        os.rename(src=file, dst=subpath+'/%d-%d.bmp' % (int(folder), file_idx))

