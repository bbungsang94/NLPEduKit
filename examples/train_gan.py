# Attributes to Images using CelebA
import os
from typing import Optional
from tqdm import tqdm
import clip
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from creadtonlp.models.gan import Generator, Discriminator

def test(generator_path: str, prompt: str, save_path: str = "./"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(512, [3, 64, 64]).to(device)
    gen_weight = torch.load(generator_path)
    generator.load_state_dict(gen_weight)
    generator.eval()
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    inv_trans = transforms.Compose([transforms.Normalize([0], [1/0.5]),
                                   transforms.Normalize([-0.5], [1])])
    
    with torch.no_grad():
        text_tokens = clip.tokenize(prompt).to(device)
        text_embeddings = clip_model.encode_text(text_tokens).to(torch.float32)
        sample_image = inv_trans(generator(text_embeddings)).squeeze().detach().cpu()
        sample_image = sample_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        pil_image = Image.fromarray(sample_image)
        pil_image.save(os.path.join(save_path, "inference_image.png"))
        
def train(n_epochs=20000, batch_size=8, save_period=5, lr_g=0.0002, betas_g=(0.5, 0.999), lr_d=0.0002, betas_d=(0.5, 0.999)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, sample = get_loader(batch_size=batch_size, device=device)
    sample_emb, real_img = sample
    
    generator = Generator(sample_emb.shape[-1], real_img.shape[1:]).to(device)
    discriminator = Discriminator(real_img.shape[1:]).to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=betas_g)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas_d)

    adversarial_loss = nn.BCELoss()
    for epoch in range(n_epochs):
        for sample in tqdm(loader):
            sample_emb, real_img = sample
            batch_size = sample_emb.shape[0]
            fake_img = generator(sample_emb)
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optimizer_D.zero_grad()
    
            real_loss = adversarial_loss(discriminator(real_img), real_labels)
            fake_loss = adversarial_loss(discriminator(fake_img.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            g_loss = adversarial_loss(discriminator(fake_img), real_labels)
            g_loss.backward()
            optimizer_G.step()
            
        if epoch % save_period == 0:
            print(f"Epoch [{epoch}/{n_epochs}] Loss D: {d_loss.item()}, loss G: {g_loss.item()}")
            save_root = os.path.join("../outputs/cgan-celeba/%06d" % epoch)
            if os.path.exists(save_root) is False:
                os.mkdir(save_root)
            save_samples(save_root, real_img, fake_img)
            torch.save(generator.state_dict(), os.path.join(save_root, "generator_epoch_%06d.pth" % epoch))
            torch.save(discriminator.state_dict(), os.path.join(save_root, "discriminator_epoch_%06d.pth" % epoch))

def save_samples(path: str, *images):
    from torchvision.utils import make_grid
    inv_trans = transforms.Compose([transforms.Normalize([0], [1/0.5]),
                                   transforms.Normalize([-0.5], [1])])

    for index, image in enumerate(images):
        grid = make_grid(inv_trans(image.detach().cpu()), nrow=3)
        grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(grid)
        im.save(os.path.join(path, "%d-th grid_image.png" % (index)))


def get_loader(batch_size: int, device: Optional[torch.device | str]):
    # CelebA dataset loading
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = CelebADataset(attr_path=r"../datasets/celeba-dataset/list_attr_celeba.csv",
                            img_root=r"../datasets/celeba-dataset/img_align_celeba/img_align_celeba",
                            transform=transform, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sample = next(iter(dataloader))
    return dataloader, sample


class CelebADataset(Dataset):
    def __init__(self, attr_path: str, img_root: str, transform=None, device="cuda"):
        self.device = torch.device(device)
        self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=device)
        
        whole_frame = pd.read_csv(attr_path)
        attr_frame = whole_frame.drop(columns=['image_id'])
        self.x = []
        for index, row in attr_frame.iterrows():
            print(index)
            attrs = []
            for col in attr_frame.columns:
                if row[col] >= 0:
                    attrs.append(col.replace('_', ' '))
            attrs = ", ".join(attrs)
            embeddings = self.get_embeddings(attrs, model=self.clip_model)
            self.x.append(embeddings.squeeze().cpu().detach().clone())
            
        self.y = []
        image_series = whole_frame['image_id']
        for value in tqdm(image_series):
            full_path = os.path.join(img_root, value)
            img = Image.open(full_path)
            img = transform(img) if transform is not None else img
            self.y.append(img)
       
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index].to(self.device), self.y[index].to(self.device)

    def get_embeddings(self, texts, model, device="cuda"):
        text_tokens = clip.tokenize(texts).to(device)
        text_embeddings = model.encode_text(text_tokens)
        return text_embeddings.to(torch.float32)

if __name__ == "__main__":
    # train()
    test(generator_path=r"./outputs/cgan-celeba/000250/generator_epoch_000250.pth",
         prompt=["Bangs, Blond Hair, Chubby, Arched Eyebrows, Sideburns, Wavy Hair, Wearing Necklace"],
         save_path="./")