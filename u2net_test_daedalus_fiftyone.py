# tutorial https://medium.com/voxel51/loading-open-images-v6-and-custom-datasets-with-fiftyone-18b5334851c3
import fiftyone as fo
import os
from PIL import Image
import numpy as np

dataset = fo.Dataset("open_images_v7")
dataset.persistent = False

name = "Daedalus"
dataset_dir = "/home/supreme/datasets-nas/INAF/daedalus/dataset-pairs/test"
prediction_path = "/home/supreme/nic/U-2-Net/test_data/u2net_results_daedalus"
hr = 'image-HR'
lr = 'image-LR'
mask = 'mask'
with fo.ProgressBar() as pb:
    for image_name in os.listdir(prediction_path):
        image_path = os.path.join(dataset_dir, lr, image_name)
        sample = fo.Sample(filepath=image_path)

        correct_mask = Image.open(os.path.join(dataset_dir, mask, image_name)).convert('L')
        correct_mask = np.asarray(correct_mask)
        
        sample["segmentations"] = fo.Segmentation(
            mask=correct_mask,
        )

        pred_mask = Image.open(os.path.join(prediction_path, image_name)).convert('L')
        pred_mask = np.asarray(pred_mask)
        sample["prediction"] = fo.Segmentation(
            mask=pred_mask,
        )

        dataset.add_sample(sample)


if __name__ == "__main__":
    session = fo.launch_app(dataset)
    session.wait()
