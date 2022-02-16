import os
from skimage import io, transform
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import ToTensorLab
from daedalus_dataset import DaedalusDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from ignite import metrics
from tqdm import tqdm

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


def main():
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    model_dir = 'saved_models/u2net_daedalus1/'
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = DaedalusDataset(
        '/home/supreme/datasets-nas/INAF/daedalus/dataset-pairs/train',
        transform=transforms.Compose(
            [
                ToTensorLab(flag=0)
            ]
        )
    )
    batch_size = 10
    img_name_list = test_salobj_dataset.hr_img_name_list
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    models_names = os.listdir(model_dir)
    models_names.sort()
    results = []
    for models_name in models_names:
        print(models_name)
        model_path = os.path.join(model_dir, models_name)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path, map_location='cpu'))
        net.eval()

        # --------- 4. inference for each image ---------
        loss_avg = metrics.Average()
        ssim = metrics.SSIM(data_range=1.0)
        psnr = metrics.PSNR(data_range=1.0)
        for i_test, data_test in enumerate(test_salobj_dataloader):
            # print("\tinferencing:",img_name_list[i_test].split(os.sep)[-1])
            
            inputs, labels = data_test['image'], data_test['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            with torch.no_grad():
                d0, d1,d2,d3,d4,d5,d6 = net(inputs_v)
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            loss_avg.update(loss)
            if labels_v.shape[0] == batch_size:
                ssim.update((labels_v, d0))
                psnr.update((labels_v, d0))
        loss_avg_comp = loss_avg.compute().item()
        ssim_avg = ssim.compute().item()
        psnr_avg = psnr.compute().item()
        print('\tLoss Avg:', loss_avg_comp, 'SSIM:', ssim_avg, 'PSNR:', psnr_avg)
        results.append([models_name, ssim_avg])
    results.sort(key=lambda x: x[1])
    print(results[:3])
    

if __name__ == "__main__":
    main()
