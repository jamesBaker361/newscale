from torch.utils.data import Dataset
import os
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import random
from datasets import load_dataset
import datasets
import torchvision
import json
from collections import defaultdict



class MaskDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.img_list=[]
        self.cat_list=[]
        self.mask_list=[]
        self.image_processor=VaeImageProcessor()
        mask_dir=os.path.join("data","datasets","gt_keep_masks")
        for sub in ["thick","thin"]: #:
            subdir=os.path.join(mask_dir,sub)
            for file in   os.listdir(subdir):
                self.mask_list.append(os.path.join(subdir,file))
                
    def __len__(self):
        return len(self.img_list)
    
    def get_mask(self):
        mask=  Image.open(self.mask_list[random.randint(0,len(self.mask_list)-1)]).convert("L").resize((self.dim,self.dim))
        mask=self.image_processor.pil_to_numpy(mask)
        mask=self.image_processor.numpy_to_pt(mask)
        return mask #we DONT want to normalize so we do this
        
        
class CIFAR10Dataset(MaskDataset):   
    def __init__(self, split="train", dim=256, limit_per_class=150):
        super().__init__()
        self.dim = dim

        self.data = load_dataset("uoft-cs/cifar10", split=split)
        self.data = self.data.cast_column("image", datasets.Image())

        class_mapping = json.load(open("sun397_class_index.json"))

        self.indices = []
        self.cat_list = []
        self.label_count = defaultdict(int)

        for i, row in enumerate(self.data):
            label = row["label"]

            if limit_per_class>0 and self.label_count[label] >= limit_per_class:
                continue

            self.indices.append(i)
            self.cat_list.append(label)
            self.label_count[label] += 1

        self.cat_set = set(self.cat_list)

    def __getitem__(self, idx):
        row = self.data[self.indices[idx]]
        image = row["image"].resize((self.dim, self.dim))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": self.image_processor.preprocess([image])[0],
            "caption": self.cat_list[idx],
            "mask": self.get_mask(),
        }

    def __len__(self):
        return len(self.indices)   
    
    
class CIFAR100Dataset(MaskDataset):   
    def __init__(self, split="train", dim=256, limit_per_class=150):
        super().__init__()
        self.dim = dim

        self.data = load_dataset("uoft-cs/cifar100", split=split)
        self.data = self.data.cast_column("image", datasets.Image())

        class_mapping = json.load(open("sun397_class_index.json"))

        self.indices = []
        self.cat_list = []
        self.label_count = defaultdict(int)

        for i, row in enumerate(self.data):
            label = row["fine_label"]

            if limit_per_class>0 and self.label_count[label] >= limit_per_class:
                continue

            self.indices.append(i)
            self.cat_list.append(label)
            self.label_count[label] += 1

        self.cat_set = set(self.cat_list)

    def __getitem__(self, idx):
        row = self.data[self.indices[idx]]
        image = row["img"].resize((self.dim, self.dim))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": self.image_processor.preprocess([image])[0],
            "caption": self.cat_list[idx],
            "mask": self.get_mask(),
        }

    def __len__(self):
        return len(self.indices)         
        
class FFHQDataset(MaskDataset):
    def __init__(self,dim:int=256,split:str="train"):
        super().__init__()
        self.dim=dim
        with open("ffhq_lite.json","r") as file:
            lite_json=json.load(file) #{'training', 'validation'}
        new_lite_json={
            k:set(v) for k,v in lite_json.items()
        }
        key={"train":"training","test":"validation"}[split]
        base_dir="images1024x1024"
        for subdir in os.listdir(base_dir):
            subdir_path=os.path.join(base_dir,subdir)
            if os.path.isdir(subdir_path):
                
                for img in os.listdir(subdir_path):
                    if img.endswith("png"):
                        img_path=os.path.join(subdir_path,img)
                        if img_path in new_lite_json[key]:
                            self.img_list.append(os.path.join(subdir_path,img))
                            self.cat_list.append("face")
        self.cat_set=set(self.cat_list)
        
        
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess([Image.open(self.img_list[index]).resize((self.dim,self.dim))])[0],
            "caption":self.cat_list[index],
            "mask":self.get_mask(),
            #"mask_out":Image.open(self.mask_list[random.randint(0,len(self.mask_list-1))]).convert("L").resize((self.dim,self.dim))
        }


class MiniImageNet(MaskDataset):
    def __init__(self, split="train", dim=256, limit_per_class=-1):
        super().__init__()
        self.dim = dim

        self.data = load_dataset("timm/mini-imagenet", split=split)
        self.data = self.data.cast_column("image", datasets.Image())

        class_index = json.load(open("imagenet_class_index.json"))
        class_mapping = {k: v[1] for k, v in class_index.items()}

        self.indices = []
        self.cat_list = []
        self.label_count = defaultdict(int)

        for i, row in enumerate(self.data):
            label = class_mapping[str(row["label"])]

            if limit_per_class > 0 and self.label_count[label] >= limit_per_class:
                continue

            self.indices.append(i)
            self.cat_list.append(label)
            self.label_count[label] += 1

        self.cat_set = set(self.cat_list)

    def __getitem__(self, idx):
        row = self.data[self.indices[idx]]

        img = row["image"]

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((self.dim, self.dim), Image.BILINEAR)

        return {
            "image": self.image_processor.preprocess([img])[0],
            "caption": self.cat_list[idx],
            "mask": self.get_mask(),
        }

    def __len__(self):
        return len(self.indices)
        
        
class SUNDataset(MaskDataset):
    def __init__(self, split="train", dim=256, limit_per_class=150):
        super().__init__()
        self.dim = dim

        self.data = load_dataset("tanganke/sun397", split=split)
        self.data = self.data.cast_column("image", datasets.Image())

        class_mapping = json.load(open("sun397_class_index.json"))

        self.indices = []
        self.cat_list = []
        self.label_count = defaultdict(int)

        for i, row in enumerate(self.data):
            label = class_mapping[str(row["label"])]

            if self.label_count[label] >= limit_per_class:
                continue

            self.indices.append(i)
            self.cat_list.append(label)
            self.label_count[label] += 1

        self.cat_set = set(self.cat_list)

    def __getitem__(self, idx):
        row = self.data[self.indices[idx]]
        image = row["image"].resize((self.dim, self.dim))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": self.image_processor.preprocess([image])[0],
            "caption": self.cat_list[idx],
            "mask": self.get_mask(),
        }

    def __len__(self):
        return len(self.indices)

        

class AFHQDataset(MaskDataset):
    
    def __init__(self,split:str="train",categories:list=["cat","dog","wild"],dim:int=256):
        super().__init__()
        self.dim=dim
        self.split=split
        self.categories=categories
        
        for cat in categories:
            dir_path=os.path.join("data","afhq",split,cat)
            for file in os.listdir(dir_path):
                if file.endswith("jpg") or file.endswith("png"):
                    self.img_list.append(os.path.join(dir_path,file))
                    self.cat_list.append(cat)

        self.cat_set=set(self.cat_list)
                
        

    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess([Image.open(self.img_list[index]).resize((self.dim,self.dim))])[0],
            "caption":self.cat_list[index],
            "mask":self.get_mask(),
            #"mask_out":Image.open(self.mask_list[random.randint(0,len(self.mask_list-1))]).convert("L").resize((self.dim,self.dim))
        }