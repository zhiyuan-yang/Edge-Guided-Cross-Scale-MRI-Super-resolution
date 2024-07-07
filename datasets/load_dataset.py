from torch.utils import data
import numpy as np
import glob
import os 
from torchvision import transforms
import cv2
    
    
class RefSRDataset():
    def __init__(self,lr_path, hr_path, ref_path, mode='test',transform_t=transforms.ToTensor()):
        self.transform_t = transform_t
        self.mode = mode
        
        ref_data_path = sorted(glob.glob(os.path.join(ref_path,"*_*.npy")))
        lr_data_path =sorted(glob.glob(os.path.join(lr_path,"*_*.npy")))
        hr_data_path =sorted(glob.glob(os.path.join(hr_path,"*_*.npy")))
        
        ref_data = np.array([np.load(frame) for frame in ref_data_path]).astype(np.float32)
        lr_data = np.array([np.load(frame) for frame in lr_data_path]).astype(np.float32)
        hr_data = np.array([np.load(frame) for frame in hr_data_path]).astype(np.float32)
        
        self.ref_data = ref_data
        self.lr_data = lr_data
        self.hr_data = hr_data
        
    
    def __getitem__(self, index):
        lr = self.lr_data[index]
        hr = self.hr_data[index]
        ref = self.ref_data[index]
        
        lr = (lr - lr.min())/(lr.max() - lr.min())
        hr = (hr - hr.min())/(hr.max() - hr.min())
        ref = (ref -ref.min())/(ref.max() - ref.min())
        
        lr_sx = cv2.Sobel(lr, cv2.CV_32F, 1, 0)
        lr_sy  = cv2.Sobel(lr, cv2.CV_32F, 0, 1)
        lr_s = cv2.addWeighted(lr_sx, 0.5, lr_sy, 0.5, 0)
        
        hr_sx = cv2.Sobel(hr, cv2.CV_32F, 1, 0)
        hr_sy  = cv2.Sobel(hr, cv2.CV_32F, 0, 1)
        hr_s = cv2.addWeighted(hr_sx, 0.5, hr_sy, 0.5, 0)
        
        lr = self.transform_t(lr.copy())
        hr = self.transform_t(hr.copy())
        ref = self.transform_t(ref.copy())
        lr_s = self.transform_t(lr_s.copy())
        hr_s = self.transform_t(hr_s.copy())
        
        return {"LR":lr,"HR":hr, "Ref":ref, "LR_s":lr_s, "HR_s":hr_s}


    def __len__(self):
        return len(self.hr_data)
              