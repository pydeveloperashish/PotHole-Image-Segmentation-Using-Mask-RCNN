import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

SAMPLES_DIR = 'Sample/balloon'

DATA_DIR = 'PotHole'

# Directory to save logs and trained model
ROOT_DIR = 'logs'

#!git clone https://www.github.com/matterport/Mask_RCNN.git ../Mask_RCNN
#os.chdir('../Mask_RCNN')
#!python setup.py -q install


print(os.path.join(ROOT_DIR, ''))
sys.path.append(os.path.join(ROOT_DIR, ''))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# !wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
 #!ls -lh mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = r'D:\Python37\Projects\models\research\object_detection\mask_rcnn_inception_v2_coco_2018_01_28\Mask_RCNN-master\mask_rcnn_tumor_detector_0015.h5'


# print(os.listdir("./"))
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs") #/kaggle/working/logs
DEFAULT_LOGS_DIR = 'logs'
# print(os.path.join(ROOT_DIR, "logs"))

class TumorConfig(Config):
    """Configuration for training on the brain tumor dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'tumor_detector'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + tumor
    DETECTION_MIN_CONFIDENCE = 0.85    
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001
    
config = TumorConfig()
config.display()



    
class BrainScanDataset(utils.Dataset):

    def load_brain_scan(self, dataset_dir, subset):
        """Load a subset of the FarmCow dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("tumor", 1, "tumor")

        # Train or validation dataset?
        assert subset in ["train", "val", 'test']
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(DATASET_DIR, subset, 'annotations_'+subset+'.json')))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "tumor",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, 
                height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a farm_cow dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tumor":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
'''


model = modellib.MaskRCNN(
    mode='training', 
    config=config, 
    model_dir=DEFAULT_LOGS_DIR
)

model.load_weights(
    COCO_WEIGHTS_PATH, 
    by_name=True, 
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)
'''
def train(model):
    """Train the model."""
    # Training dataset.
    # Training dataset.
    dataset_train = BrainScanDataset()
    dataset_train.load_brain_scan(DATASET_DIR, 'train')
    dataset_train.prepare()

# Validation dataset
    dataset_val = BrainScanDataset()
    dataset_val.load_brain_scan(DATASET_DIR, 'val')
    dataset_val.prepare()

    dataset_test = BrainScanDataset()
    dataset_test.load_brain_scan(DATASET_DIR, 'test')
    dataset_test.prepare()

# Since we're using a very small dataset, and starting from
# COCO trained weights, we don't need to train too long. Also,
# no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(
        dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=15,
        layers='heads'
    )
    
#     model.save('64x3-CNN.model')

# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) #



if __name__ == '__main__':
    print('Train')
    
    config = tumorConfig()
    
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)
    
    weights_path = COCO_WEIGHTS_PATH
#     COCO_WEIGHTS_PATH = '../../../mask_rcnn_coco.h5'
    
    # Find last trained weights
    # weights_path = model.find_last()[1]
    
    
    model.load_weights(COCO_WEIGHTS_PATH)
    #, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    train(model)



# remove files to allow committing (hit files limit otherwise)
#!rm -rf /kaggle/working/Mask_RCNN


# from IPython.display import HTML

# def create_download_link(title = "Download CSV file", filename = "data.csv"):  
#     html = '<a href={filename}>{title}</a>'
#     html = html.format(title=title,filename=filename)
#     return HTML(html)

# # create a link to download the dataframe which was saved with .to_csv method
# # create_download_link(filename='/kaggle/working/logs/balloon20190325T1747/mask_rcnn_balloon_0001.h5')
# create_download_link(filename='balloon20190326T1301/mask_rcnn_balloon_0001.h5')

