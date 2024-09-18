# Attributes to Images using CelebA
import os
import clip
import torch
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_processor = clip.load("ViT-B/32", device=device)
    loader, sample = get_loader(batch_size=8)


def get_embeddings(texts, model, device="cuda"):
    text_tokens = clip.tokenize(texts).to(device)
    text_embeddings = model.encode_text(text_tokens)
    return text_embeddings


def get_loader(batch_size: int):
    from torchvision import transforms
    # CelebA dataset loading
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = CelebADataset(attr_path=r"../datasets/celeba-dataset/list_attr_celeba.csv",
                            img_root=r"../datasets/celeba-dataset/img_align_celeba/img_align_celeba",
                            transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample = next(iter(dataloader))
    return dataloader, sample


class CelebADataset(Dataset):
    def __init__(self, attr_path: str, img_root: str, transform=None):
        whole_frame = pd.read_csv(attr_path)
        self.y = []
        image_series = whole_frame['image_id']
        for value in image_series:
            full_path = os.path.join(img_root, value)
            img = Image.open(full_path)
            img = transform(img) if transform is not None else img
            self.y.append(img)

        attr_frame = whole_frame.drop(columns=['image_id'])
        self.x = []
        for index, row in attr_frame.iterrows():
            attrs = []
            for col in attr_frame.columns:
                if row[col] >= 0:
                    attrs.append(col.replace('_', ' '))
            self.x.append(attrs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    train()