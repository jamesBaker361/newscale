from torch.utils.data import Dataset
import os
from PIL import Image

class AFHQDataset(Dataset):
    
    def __init__(self,split:str="train",categories:list=["cat","dog","wild"]):
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
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        return {
            "image":Image.open(self.img_list[index]),
            "cat":self.cat_list[index]
        }