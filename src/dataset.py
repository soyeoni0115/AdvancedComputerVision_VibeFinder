from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CafeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for cafe in os.listdir(root_dir):
            cafe_path = os.path.join(root_dir, cafe)
            for img in os.listdir(cafe_path):
                self.samples.append((
                    os.path.join(cafe_path, img),
                    int(cafe)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label