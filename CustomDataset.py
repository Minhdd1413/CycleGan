# Import
import os
import glob # lấy danh sách vào tên thư viện theo điều kiện
import random
import PIL.Image as Img

from torch.utils.data import Dataset

class ImageDataset(Dataset): # Dataset template 
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        # Lấy danh sách tên file và thư mục theo điều kiện: glob.glob(pattern, *, recursive=False)
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.*"))

    def __getitem__(self, index):
        item_A = self.transform(Img.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Img.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Img.open(self.files_B[index % len(self.files_B)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
