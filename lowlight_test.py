import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time



def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    data_lowlight = Image.open(image_path)

 

    data_lowlight = (np.asarray(data_lowlight)/255.0)


    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net = DCE_net.eval()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
    start = time.time()
    enhanced_image,_,r,g,b,_ = DCE_net(data_lowlight)
    #r1,r2 = torch.split(r, 3, dim=1)
    rx = r
    #rx = 0.5*(rx+1)
    end_time = (time.time() - start)
    print(end_time,g,b)
    image_path = image_path.replace('test_data','result')
    result_path = image_path
    if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
        os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

    torchvision.utils.save_image(enhanced_image, result_path)
    image_path1 = image_path.replace('result','mengban')
    result_path1 = image_path1
    if not os.path.exists(image_path1.replace('/'+image_path1.split("/")[-1],'')):
        os.makedirs(image_path1.replace('/'+image_path1.split("/")[-1],''))
    torchvision.utils.save_image(rx, result_path1)

if __name__ == '__main__':
# test_images
    with torch.no_grad():
        filePath = 'data/test_data/'
    
        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath+file_name+"/*") 
            for image in test_list:
                # image = image
                print(image)
                lowlight(image)

        

