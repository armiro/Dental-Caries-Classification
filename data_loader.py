import numpy as np, cv2.cv2 as cv2, glob, matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

path = './opg_images_labeled/healthy/*'
neg_images = list()
for img_name in glob.glob(pathname=path):
    img = load_img(path=img_name, color_mode='grayscale')
    img = img_to_array(img=img, data_format='channels_last')
    neg_images.append(img)

neg_images = np.array(neg_images)
print('num healthy images:', len(neg_images))

path = './opg_images_labeled/carious/*'
pos_images = list()
for img_name in glob.glob(pathname=path):
    img = load_img(path=img_name, color_mode='grayscale')
    img = img_to_array(img=img, data_format='channels_last')
    pos_images.append(img)

pos_images = np.array(pos_images)
print('num carious images:', len(pos_images))

neg_labels = [0 for _ in range(len(neg_images))]
pos_labels = [1 for _ in range(len(pos_images))]

X = np.concatenate((pos_images, neg_images))
y = np.array(pos_labels + neg_labels)

# pad the asymmetric images to have them all as squared
X_padded = list()
for image in X:
    height, width = image.shape[:2]
    if width >= height:
        img_pd = np.pad(image.squeeze(), pad_width=((int((width-height)/2), int((width-height)/2)),(0, 0)), mode='constant')
    else:
        img_pd = np.pad(image.squeeze(), pad_width=((0, 0),(int((height-width)/2), int((height-width)/2))), mode='constant')
    X_padded.append(img_pd)
X = np.array(X_padded)

# resize images
X = np.array([cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_CUBIC) for image in X])
X = np.array([np.expand_dims(a=image, axis=-1) for image in X])
X = X.astype(dtype=np.uint8)

rnd_idx = np.random.choice(a=len(X), size=None)
plt.imshow(X=X[rnd_idx].squeeze(), cmap='gray')
plt.title('a random image from the dataset')
plt.show()

np.save(file='./opg_v3_x.npy', arr=X)
np.save(file='./opg_v3_y.npy', arr=y)
