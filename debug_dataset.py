from torchvision import transforms
from torch.utils.data import DataLoader


from daedalus_dataset import DaedalusDataset
from data_loader import ToTensorLab

salobj_dataset = DaedalusDataset(
    '/home/supreme/datasets-nas/INAF/daedalus/dataset-pairs/train',
    transform=transforms.Compose(
        [
            ToTensorLab(flag=0)
        ]
    )
)
salobj_dataloader = DataLoader(
    salobj_dataset, 
    batch_size=2, 
    shuffle=False, 
    num_workers=1
)


for i, x in enumerate(salobj_dataloader):
    #print(i, x["imidx"].shape, x["image"].shape, x["label"].shape)
    # print(i, x["label"].dtype)
    print(i)