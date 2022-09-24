import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

from tqdm import tqdm
from CustomDataset import ImageDataset  # Xử lý testset
from Network import Generator # G network

# Hyperparameter
cuda = "store_true"
image_size = 256
cudnn.benchmark = True

# Make result folder
try:
    os.makedirs(args.outf)
except OSError:
    pass

# Set dir
data_in_dir = "./data/horse2zebra" # Real data
pre_train_dir = "./Pre_train" # Pre-train

# Setup device
if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset
dataset = ImageDataset(root=data_in_dir,
                       transform=transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ]),
                       mode="test")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

device = torch.device("cuda:0" if cuda else "cpu")

# create model
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(pre_train_dir + "netG_A2B.pth"))
netG_B2A.load_state_dict(torch.load(pre_train_dir + "netG_B2A.pth"))

# Set model mode
netG_A2B.eval()
netG_B2A.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

for i, data in progress_bar:
    # get batch size data
    real_images_A = data["A"].to(device)
    real_images_B = data["B"].to(device)

    # Generate output
    fake_image_A = 0.5 * (netG_B2A(real_images_B).data + 1.0)
    fake_image_B = 0.5 * (netG_A2B(real_images_A).data + 1.0)

    # Save image files
    vutils.save_image(fake_image_A.detach(), f"./Result/A/{i + 1:04d}.png", normalize=True)
    vutils.save_image(fake_image_B.detach(), f"./Result/B/{i + 1:04d}.png", normalize=True)

    progress_bar.set_description(f"Process images {i + 1} of {len(dataloader)}")
