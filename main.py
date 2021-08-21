
from network import Checkpointer
from network import UNET
from data_loader import Dataloader

import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt


seed = 42
np.random.seed = seed

project_dir = 'C:\\Users\\ASUS\\Desktop\\unets'
img_width = 128
img_height = 128
img_channels = 3
test_path = 'test/'
train_path = 'train/'

# loading the data and preprocessing
dataloader = Dataloader(project_dir, img_width, img_height,
                        img_channels, train_path, test_path)
dataloader.preprocess()
x_train = dataloader.x_train
y_train = dataloader.y_train
x_test = dataloader.x_test


# testing an image
imshow(x_train[1])
plt.show()
imshow(y_train[1])
plt.show()


# model architecture
model = UNET(128, 128, 3)
model.build()

# getting the summary()
model.summary()

# checkpointing
checkpointer = Checkpointer("nuclie_model.py", 'new_logs')
checkpointer.checkpnt()
callbacks = checkpointer.callbacks()

# model fitting
model.fit(x_train, y_train, callbacks)
model.save('nucliet_segmentation.h5')


# predictions
test_predictions = model.predict(x_test)

# tensorboard command
# !tensorboard --logdir=new_logs/ --host localhost --port 8088
