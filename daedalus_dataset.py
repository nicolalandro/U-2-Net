import os
from torch.utils.data import Dataset
from skimage import io
import numpy as np

class DaedalusDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path

        self.hr_img_name_list = os.listdir(
            os.path.join(self.dataset_path, 'image-HR'))
        self.hr_img_name_list.sort()

        self.lr_img_name_list = os.listdir(
            os.path.join(self.dataset_path, 'image-LR'))
        self.lr_img_name_list.sort()

        self.label_name_list = os.listdir(
            os.path.join(self.dataset_path, 'mask'))
        self.label_name_list.sort()

        self.transform = transform

    def __len__(self):
        return len(self.hr_img_name_list)

    def __getitem__(self, idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])
        hr_path = os.path.join(self.dataset_path, 'image-HR', self.hr_img_name_list[idx])
        hr_image = io.imread(hr_path)

        lr_path = os.path.join(self.dataset_path, 'image-LR', self.lr_img_name_list[idx])
        lr_image = io.imread(lr_path)

        empty_channel = np.zeros(lr_image.shape)
        
        image = np.stack((hr_image, lr_image, empty_channel), axis=2)
        
        imidx = np.array([idx])

        mask_path = os.path.join(self.dataset_path, 'mask', self.label_name_list[idx])
        if(0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(mask_path)

        label = np.zeros(label_3.shape[0:2])
        if(3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif(2 == len(label_3.shape)):
            label = label_3

        if(3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif(2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    from data_loader import ToTensorLab
    from torchvision import transforms

    dataset = DaedalusDataset(
        '../daedalus-preprocessing/dataset-pairs',
        transform=transforms.Compose(
        [
            ToTensorLab(flag=0)
        ]
    )
    )
    sample = dataset.__getitem__(0)
    print(sample['image'].shape, sample['label'].shape)
