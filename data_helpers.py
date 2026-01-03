from torch.utils.data import Dataset
import os
from PIL import Image
import random

class AFHQDataset(Dataset):
    
    def __init__(self,split:str="train",categories:list=["cat","dog","wild"],dim:int=256):
        super().__init__()
        self.split=split
        self.categories=categories
        self.img_list=[]
        self.cat_list=[]
        for cat in categories:
            dir_path=os.path.join("data","afhq",split,cat)
            for file in os.listdir(dir_path):
                if file.endswith("jpg") or file.endswith("png"):
                    self.img_list.append(os.path.join(dir_path,file))
                    self.cat_list.append(cat)
        self.mask_list=[]
        mask_dir=os.path.join("data","datasets","gt_keep_masks")
        for sub in os.listdir(mask_dir):
            subdir=os.path.join(mask_dir,sub)
            for file in  ["thick","thin"]: # os.listdir(subdir):
                self.mask_list.append(os.path.join(subdir,file))
                '''if subdir not in ["thick","thin"]: #so i guess all the other ones should just be their own thing???
                    break'''
                
        
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        return {
            "image":Image.open(self.img_list[index]).resize((self.dim,self.dim)),
            "caption":self.cat_list[index],
            "mask":Image.open(self.mask_list[random.randint(0,len(self.mask_list-1))]).convert("L").resize((self.dim,self.dim)),
            #"mask_out":Image.open(self.mask_list[random.randint(0,len(self.mask_list-1))]).convert("L").resize((self.dim,self.dim))
        }