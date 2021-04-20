import torch
import torchvision
from network import *
from dataset import *
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import tqdm
from torchvision.utils import save_image
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def level_gen(model,ngf,channel,trap_idx):
    generator=model(ngf,channel,trap_idx)
    generator.to(device)
    generator.load_state_dict(torch.load('saved_models_final/WGAN-GP/generator_stained199.pth'))
    return generator

def create_model(model,ngf,channel):
    gen_list=[]
    for i in range(6):
        gen_list.append(model(ngf,channel,i))
    return gen_list



transforms_ = [
    transforms.Resize((int(256), int(256))),
    # transformsRandomCrop((256,256)),
    # RandomFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
        GenerateDataset('finaldataset', transforms_=transforms_),

        batch_size=1,
        shuffle=True,
        num_workers=2
    )
dataloader_val = DataLoader(
    GenerateDataset('finaldataset', mode='test', transforms_=transforms_),
    batch_size=1,
    shuffle=False,
    num_workers=2
)

os.makedirs('trap/',exist_ok=True)
data=next(iter(dataloader))
gen_list=create_model(TrapTRGenerator,128,512)
for i,gen in enumerate(gen_list):
    gen=gen.to(device)
    input_unstained=data['unstained'].to(device)
    generate_samples=gen(input_unstained)
    save_genearte = Image.fromarray(
        generate_samples[0].mul(0.5).add_(0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                             torch.uint8).numpy())
    save_genearte.save('trap/%d.jpg'%i)



