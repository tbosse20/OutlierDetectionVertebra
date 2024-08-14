import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class CTScanDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.split = split
        self.img_dir = os.path.join(data_dir,'crops')
        self.df_dir = os.path.join(data_dir,'dist_fields')
        self.crops = os.listdir(self.img_dir)
        self.crops = [x for x in self.crops if 'label' not in x] #No labels
        self.dist_fields = os.listdir(self.df_dir)
        
        self.training_id_list_file = "./challenge_results/custom_train_list_436.txt"
        self.train_scan_ids = np.loadtxt(str(self.training_id_list_file), delimiter=",", dtype=str)
        if self.split == "train":
            self.crops = list(filter(lambda x: os.path.basename(x)[:11] in self.train_scan_ids, self.crops))
            self.dist_fields = list(filter(lambda x: os.path.basename(x)[:11] in self.train_scan_ids, self.dist_fields))
        elif self.split == "val":
            self.crops = list(filter(lambda x: os.path.basename(x)[:11] not in self.train_scan_ids, self.crops))
            self.dist_fields = list(filter(lambda x: os.path.basename(x)[:11] not in self.train_scan_ids, self.dist_fields))
        else:
            raise ValueError
        self.crops = np.sort(self.crops)
        self.dist_fields = np.sort(self.dist_fields)
        
        outliers = []
        
        for i in range(len(self.crops)):
            if 'outlier' in self.crops[i]:
                outliers.append(1)
            else:
                outliers.append(0)
            
        self.outliers = np.array(outliers)
        
        # self.normal = [x for x in self.crops if x.endswith('crop.nii.gz')] #crop_warp_outlier
        # self.warp = [x for x in self.crops if x.endswith('crop_warp_outlier.nii.gz')] #crop_warp_outlier
        # self.sphere_water = [x for x in self.crops if x.endswith('crop_sphere_outlier_water.nii.gz')] #crop_warp_outlier
        # self.sphere_std = [x for x in self.crops if x.endswith('crop_sphere_outlier_mean_std_inpaint.nii.gz')] #crop_warp_outlier
        
        # self.labels = labels
        # self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.crops[idx])
        dist_path = os.path.join(self.df_dir,self.dist_fields[idx])
        
        #Image
        img = nib.load(img_path)
        img = np.asanyarray(img.dataobj, dtype=np.float32)
        #Normalise
        HU_range_normalize = [-1, 1]
        min_value = np.min(img)
        max_value = np.max(img)
        img = (HU_range_normalize[1]-HU_range_normalize[0])*(img - min_value) / (max_value - min_value + 1e-7) + HU_range_normalize[0]

        
        #Distance
        dist = nib.load(dist_path)
        dist = np.asanyarray(dist.dataobj, dtype=np.float32)
        #Normalise
        HU_range_normalize = [-1, 1]
        min_value = np.min(dist)
        max_value = np.max(dist)
        dist = (HU_range_normalize[1]-HU_range_normalize[0])*(dist - min_value) / (max_value - min_value) + HU_range_normalize[0]

        
        
        #Outlier
        outlier = self.outliers[idx]
        
        #Input tensor
        inputs = np.stack((img, dist), axis=3)
        #Reshape
        inputs = np.moveaxis(inputs, -1, 0)
        inputs = inputs.astype(np.float64)
                
        #Convert to tensor
        inputs = torch.from_numpy(inputs)
        
        return inputs, outlier
        
        # # img = nib.load(img_path)
        # # img = np.asanyarray(img.dataobj, dtype=np.float32)
        
        # # img_path = os.path.join(self.img_dir,self.normal[index])
        # # dist_path
        # # heatmap_path = os.path.join(self.heatmap_dir,self.images[index].replace('img.npy','heatmap.npy'))
        # # msk_path = os.path.join(self.msk_dir,self.images[index].replace('img.npy','msk.npy'))

        # scan = np.load(self.file_paths[idx])  # Load your 3D volume
        # label = self.labels[idx]

        # if self.transform:
        #     scan = self.transform(scan)

        # return torch.tensor(scan, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# Example file paths and labels
# file_paths = '/Volumes/T9/OutlierChallenge2024/challenge_data/train'
# labels = [0, 1, ...]  # Binary labels

# # Define your transforms
# transform = Compose([
#     ToTensor(),
#     Normalize(mean=[0.5], std=[0.5])
# ])

# dataset = CTScanDataset(file_paths, labels, transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


