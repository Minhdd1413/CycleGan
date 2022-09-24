# Import thư viện
from CustomDataset import ImageDataset  # Xử lý dataset
from Network import Generator, Discriminator # G và D network
from DecayEpochs import DecayLR # 
from torchvision import transforms
from utils import weights_init, ReplayBuffer
from tqdm import tqdm

import torchvision
import PIL.Image as Img
import torch
import itertools

# Hyperparameter settings
epochs = 200 # Loops
decay_epochs = 100 # Số vòng lặp để chạy hàm tuyến tính hoá learning rate
batch_size = 1 # mini-batch size (default: 1), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
image_size = 256 # Default
print_freq = 100 # In ra milestone trong quá trình train dưới dạng ảnh
lr = 0.0002
data_in_dir = "./CycleGAN/data/horse2zebra"
sample_out_dir = "./CycleGAN/Sample"
pre_train_dir = "./CycleGAN/Pre_train/horse2zebra"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Loading data
dataset = ImageDataset(root=data_in_dir,
                       transform=transforms.Compose([
                           transforms.Resize(int(image_size * 1.12), Img.BICUBIC),
                           transforms.RandomCrop(image_size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                       unaligned=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Loading Model
netG_A_to_B = Generator().to(device)
netG_B_to_A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

netG_A_to_B.apply(weights_init)
netG_B_to_A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

# define loss function (adversarial_loss) and optimizer
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A_to_B.parameters(), netG_B_to_A.parameters()),
                               lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_lambda = DecayLR(epochs, 0, decay_epochs).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Building loop
for epoch in range(0, epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # get batch size data
        real_image_A = data["A"].to(device)
        real_image_B = data["B"].to(device)
        batch_size = real_image_A.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

        ''' ------------------------------------------- '''
        # (1) Update G network: Generators A2B and B2A
        ''' ------------------------------------------- '''

        # Set G_A and G_B's gradients to zero
        optimizer_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        identity_image_A = netG_B_to_A(real_image_A)
        loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = netG_A_to_B(real_image_B)
        loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = netG_B_to_A(real_image_B)
        fake_output_A = netD_A(fake_image_A)
        loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
        # GAN loss D_B(G_B(B))
        fake_image_B = netG_A_to_B(real_image_A)
        fake_output_B = netD_B(fake_image_B)
        loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

        # Cycle loss
        recovered_image_A = netG_B_to_A(fake_image_B)
        loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

        recovered_image_B = netG_A_to_B(fake_image_A)
        loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

        # Cộng loss and calculate gradients
        errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        # Tính gradients của G_A và G_B
        errG.backward()
        # Update G_A and G_B's weights
        optimizer_G.step()

        ''' ------------------------------------------- '''
        # (2) Update D network: Discriminator A
        ''' ------------------------------------------- '''        

        # Set D_A gradients == zero
        optimizer_D_A.zero_grad()

        # Real A image loss
        real_output_A = netD_A(real_image_A)
        errD_real_A = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
        fake_output_A = netD_A(fake_image_A.detach())
        errD_fake_A = adversarial_loss(fake_output_A, fake_label)

        # Cộng loss và tính gradients
        errD_A = (errD_real_A + errD_fake_A) / 2

        # Tính gradients D_A
        errD_A.backward()
        # Update D_A weights
        optimizer_D_A.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients == zero
        optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = netD_B(real_image_B)
        errD_real_B = adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
        fake_output_B = netD_B(fake_image_B.detach())
        errD_fake_B = adversarial_loss(fake_output_B, fake_label)

        # Cộng loss và tính gradients
        errD_B = (errD_real_B + errD_fake_B) / 2

        # Tính gradient D_B
        errD_B.backward()
        # Update D_B weights
        optimizer_D_B.step()

        progress_bar.set_description(
            f"[{epoch}/{epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(errD_A + errD_B).item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
            f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
            f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

        if i % print_freq == 0:
            torchvision.utils.save_image(real_image_A,
                              sample_out_dir + "/A" + f"/real_samples_epoch_{epoch}_{i}.png",
                              normalize=True)
            torchvision.utils.save_image(real_image_B,
                              sample_out_dir + "/B" + f"/real_samples_epoch_{epoch}_{i}.png",
                              normalize=True)

            fake_image_A = 0.5 * (netG_B_to_A(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (netG_A_to_B(real_image_A).data + 1.0)

            torchvision.utils.save_image(fake_image_A.detach(),
                              sample_out_dir + "/A" + f"/real_samples_epoch_{epoch}_{i}.png",
                              normalize=True)
            torchvision.utils.save_image(fake_image_B.detach(),
                              sample_out_dir + "/A" + f"/real_samples_epoch_{epoch}_{i}.png",
                              normalize=True)

    # Lấy điểm 
    torch.save(netG_A_to_B.state_dict(), pre_train_dir + f"/netG_A_to_B_epoch_{epoch}.pth")
    torch.save(netG_B_to_A.state_dict(), pre_train_dir + f"/netG_B_to_A_epoch_{epoch}.pth")
    torch.save(netD_A.state_dict(), pre_train_dir + f"/netD_A_epoch_{epoch}.pth")
    torch.save(netD_B.state_dict(), pre_train_dir + f"/netD_B_epoch_{epoch}.pth")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

# Save điểm cuối
torch.save(netG_A_to_B.state_dict(), pre_train_dir + f"/netG_A_to_B_epoch_{epoch}.pth")
torch.save(netG_B_to_A.state_dict(), pre_train_dir + f"/netG_B_to_A_epoch_{epoch}.pth")
torch.save(netD_A.state_dict(), pre_train_dir + f"/netD_A_epoch_{epoch}.pth")
torch.save(netD_B.state_dict(), pre_train_dir + f"/netD_B_epoch_{epoch}.pth")
