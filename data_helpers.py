from torch.utils.data import Dataset
import os
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import random
from datasets import load_dataset
import datasets
import torchvision
import json


class MaskDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.mask_list=[]
        mask_dir=os.path.join("data","datasets","gt_keep_masks")
        for sub in ["thick","thin"]: #:
            subdir=os.path.join(mask_dir,sub)
            for file in   os.listdir(subdir):
                self.mask_list.append(os.path.join(subdir,file))



class MiniImageNet(MaskDataset):
    def __init__(self,split:str="train",dim:int=256):
        super().__init__()
        self.dim=dim
        data=load_dataset("timm/mini-imagenet",split=split)
        data=data.cast_column("image",datasets.Image())
        self.image_processor=VaeImageProcessor()
        self.img_list=[]
        self.cat_list=[]
        class_index_path="imagenet_class_index.json"
        class_index=json.load(open(class_index_path,"r"))
        class_mapping={}
        for key,pair in class_index.items():
            class_mapping[key]=pair[1]
        for row in data:
            self.img_list.append(row["image"])
            self.cat_list.append(class_mapping[str(row["label"])])
            
        
            
        self.cat_set=set(self.cat_list)
            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess([self.img_list[index].resize((self.dim,self.dim))])[0],
            "caption":self.cat_list[index],
            "mask":self.image_processor.preprocess([Image.open(self.mask_list[random.randint(0,len(self.mask_list)-1)]).convert("L").resize((self.dim,self.dim))])[0],
            #"mask_out":Image.open(self.mask_list[random.randint(0,len(self.mask_list-1))]).convert("L").resize((self.dim,self.dim))
        }
        
        
        

class AFHQDataset(MaskDataset):
    
    def __init__(self,split:str="train",categories:list=["cat","dog","wild"],dim:int=256):
        super().__init__()
        self.dim=dim
        self.split=split
        self.categories=categories
        self.img_list=[]
        self.cat_list=[]
        self.image_processor=VaeImageProcessor()
        for cat in categories:
            dir_path=os.path.join("data","afhq",split,cat)
            for file in os.listdir(dir_path):
                if file.endswith("jpg") or file.endswith("png"):
                    self.img_list.append(os.path.join(dir_path,file))
                    self.cat_list.append(cat)

        self.cat_set=set(self.cat_list)
                
        
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess([Image.open(self.img_list[index]).resize((self.dim,self.dim))])[0],
            "caption":self.cat_list[index],
            "mask":self.image_processor.preprocess([Image.open(self.mask_list[random.randint(0,len(self.mask_list)-1)]).convert("L").resize((self.dim,self.dim))])[0],
            #"mask_out":Image.open(self.mask_list[random.randint(0,len(self.mask_list-1))]).convert("L").resize((self.dim,self.dim))
        }
        
if __name__=='__main__':
    data=MiniImageNet()
    for batch in data:
        break
    
    for key in batch:
        print(key, batch[key])