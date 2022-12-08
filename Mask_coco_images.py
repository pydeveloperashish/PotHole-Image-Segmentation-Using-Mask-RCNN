import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# Root directory of the project
ROOT_DIR = os.path.abspath(r"D:\Python37\Projects\models\research\object_detection\mask_rcnn_inception_v2_coco_2018_01_28\Mask_RCNN-master")

import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
from samples.coco import coco
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_coco.h5', by_name=True)

# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
image = skimage.io.imread(r'D:\Python37\Projects\models\research\object_detection\mask_rcnn_inception_v2_coco_2018_01_28\Mask_RCNN-master\images\8699757338_c3941051b6_z.jpg')

# original image
plt.figure(figsize=(12,10))
skimage.io.imshow(image)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
ac=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
#plt.savefig(r'D:\Python37\Projects\models\research\object_detection\mask_rcnn_inception_v2_coco_2018_01_28\Mask_RCNN-master\result.png')

mask = r['masks']
mask = mask.astype(int)
mask.shape
matplotlib.use('TkAgg')
i=0
for i in range(mask.shape[2]):
    temp = skimage.io.imread(r'D:\Python37\Projects\models\research\object_detection\mask_rcnn_inception_v2_coco_2018_01_28\Mask_RCNN-master\images\8699757338_c3941051b6_z.jpg')
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    plt.figure(figsize=(8,8))
    plt.savefig(r"D:\Python37\Projects\models\research\object_detection\mask_rcnn_inception_v2_coco_2018_01_28\Mask_RCNN-master\Output\%d.jpg" % (i + 1),dpi=100)
    plt.imshow(temp)
    plt.show()