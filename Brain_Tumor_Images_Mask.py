import cv2
from samples import balloonConfig
import os
import sys
import random
import math
import tensorflow as tf
#import keras
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Root directory of the project
ROOT_DIR = os.path.abspath('D:/Python37/Projects/models/research/object_detection/mask_rcnn_inception_v2_coco_2018_01_28/Mask_RCNN-master')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
#from mrcnn import visualize
import tumor_visualize_cv2
from tumor_visualize_cv2 import model, display_instances, class_names
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco"))  # To find local version
#COCO_DIR= ('C:/Python37/Projects/models/research/object_detection/mask_rcnn_inception_v2_coco_2018_01_28/Mask_RCNN-master/coco')

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = r'D:\Python37\Projects\models\research\object_detection\mask_rcnn_inception_v2_coco_2018_01_28\Mask_RCNN-master\mask_rcnn_tumor_detector_0014.h5'

# Directory of images to run detection on
IMAGE_DIR = r'D:\Python37\Projects\Brain Tumor\Brain-Tumor-Detection-master\images2'

class InferenceConfig(balloonConfig.BalloonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

#class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              # 'bus', 'train', 'truck', 'boat', 'traffic light',
               #'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               #'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               #'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            #   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
             #  'kite', 'baseball bat', 'baseball glove', 'skateboard',
              # 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               #'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             #  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
             #  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
             #  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
             #  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
             #  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             #  'teddy bear', 'hair drier', 'toothbrush']
               

CATEGORIES = ["no", "yes"]
class_names=['BG','tumor']

image=r'D:\Python37\Projects\Brain Tumor\Brain-Tumor-Detection-master\images2\aug_aug_Y92_0_1860_0_3424.jpg'


cv2_image=cv2.imread(image)

def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.
    
model_tumor = tf.keras.models.load_model(r"D:\Python37\Projects\Brain Tumor\Brain-Tumor-Detection-master\64x3-CNN-Tumor.model")
prediction = model_tumor.predict([prepare(image)])
print(CATEGORIES[int(prediction[0][0])])
img=mpimg.imread(image)
imgplot = plt.imshow(img)
plt.show()


#file_names = next(os.walk(IMAGE_DIR))[2]

#file_names=r'C:\Python37\Projects\Brain Tumor\Brain-Tumor-Detection-master\images2\aug_1 no._0_1187.jpg'
#image = skimage.io.imread(os.qpath.join(IMAGE_DIR, random.choice(file_names)))
# Run detection
results = model.detect([cv2_image], verbose=1)

# Visualize results
r = results[0]

ac=display_instances(cv2_image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
plt.imshow(ac)
plt.show()
# In[ ]:




