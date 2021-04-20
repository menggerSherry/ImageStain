import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
from network import *
from torch.backends import cudnn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, x):
        return 1.0 - max(0, x + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def sample_images(batches_done,val_dataloader,generator,save_type):
    '''
    '''
    imgs=next(iter(val_dataloader))
    real_input=imgs['unstained'].to(device)
    real_image=imgs['stained'].to(device)
    input_list=[]
    for j in range(4):
        stride = 2 ** (3 - j)
        input_list.append(real_input[..., ::stride, ::stride].to(device))
    # print(real_input.size())
    fake_image=generator(input_list)
    img_sample=torch.cat((real_input,fake_image,real_image),2)
    save_image(img_sample,'images/%s/%s.png'%(save_type,batches_done),normalize=True)


def test_image(batch_done, dataloader_val, generator, save):
    image_val = next(iter(dataloader_val))
    real_stained = image_val['stained'].to(device)
    real_unstained = image_val['unstained'].to(device)
    fake_stained = generator(real_unstained)
    real_stained_image = make_grid(real_stained, normalize=True)
    real_unstained_image = make_grid(real_unstained, normalize=True)
    fake_stained_image = make_grid(fake_stained, normalize=True)
    image_grid = torch.cat((real_unstained_image, fake_stained_image, real_stained_image), 2)
    save_image(image_grid, 'images/%s/%d.png' % (save, batch_done), normalize=True)


def main(config):
    cudnn.benchmark = True
    os.makedirs('saved_models_T/%s/' % (config.loss_type), exist_ok=True)
    os.makedirs('images_v2/%s/'%(config.loss_type), exist_ok=True)
    # dataloader = get_loader(config.path, config.train_mode, config.batch_size)
    # dataloader_val = get_loader(config.path, 'test', config.batch_size)
    transforms_ = [
        transforms.Resize((int(256), int(256))),
        # transformsRandomCrop((256,256)),
        # RandomFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        StainedDataset(config.path, transforms_=transforms_),

        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    dataloader_val = DataLoader(
        StainedDataset(config.path, mode='test', transforms_=transforms_),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )

    stain_generator = TRGenerator(128,512)
    #stain_generator = UNetS(512)
    # discriminator = FeatureDiscriminator(config.num_channels*2,config.fndf)
    pixel_discriminator = PixelDiscriminator(config.num_channel*2,config.pndf)

    stain_generator=stain_generator.to(device)
    # discriminator=discriminator.to(device)
    pixel_discriminator=pixel_discriminator.to(device)
    optimizer_G = torch.optim.Adam(stain_generator.parameters(),
                                   lr=config.lr,
                                   betas=(config.beta1, config.beta2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(),
    #                                        lr=config.lr,
    #                                        betas=(config.beta1, config.beta2))
    optimizer_D_pixel = torch.optim.Adam(pixel_discriminator.parameters(),
                                             lr=config.lr,
                                             betas=(config.beta1, config.beta2)
                                             )
    if config.loss_type=='WGAN-GP':
        criterion_gan=loss.WassersteinLoss(pixel_discriminator,stain_generator)

    elif config.loss_type=='BCE-loss':
        criterion_gan=loss.StandardGAN(pixel_discriminator,stain_generator)

    elif config.loss_type=='LS-loss':
        criterion_gan=loss.LSGAN(pixel_discriminator,stain_generator)

    elif config.loss_type=='Logistic-loss':
        criterion_gan=loss.LogisticGAN(pixel_discriminator,stain_generator)
    criterion_pixel=nn.BCELoss()


    if config.use_fm=='True':
        alpha_fm=10
    else:
        alhpa_fm=0

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step
    )
    # lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer_D, lr_lambda=LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step
    # )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_pixel, lr_lambda=LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step
    )


    if config.epoch != 0:
        stain_generator.load_state_dict(
            torch.load('saved_models_s/%s/generator_unstained%d.pth' % (config.save_checkpoint,config.epoch)))

        # discriminator.load_state_dict(
        #     torch.load('saved_models/%s/discriminator_unstained%d.pth' % (config.save_checkpoint,config.epoch)))
        pixel_discriminator.load_state_dict(
            torch.load('saved_models_s/%s/discriminator_stained%d.pth' % (config.save_checkpoint,config.epoch)))
        stain_generator.train()
        # discriminator.train()
        pixel_discriminator.train()


    dis_shape=(config.batch_size,1,32,32)
    valid=torch.ones((config.batch_size,1,256,256),device=device)
    fake=torch.zeros((config.batch_size,1,256,256),device=device)
    iters=0
    loss_dis = torch.tensor(0)
    loss_dis_pixel = torch.tensor(0)
    loss_gan=torch.tensor(0)
    for epoch in range(config.epoch, config.n_epochs):
        for i, batch in enumerate(dataloader):
            real_stained = batch['stained'].to(device)
            # print(real_stained.size())
            real_unstained = batch['unstained'].to(device)
            iters += 1
            loss_dis_pixel,loss_gan=criterion_gan.loss(real_unstained,real_stained)

            optimizer_G.zero_grad()
            loss_gan.backward()
            optimizer_G.step()

            optimizer_D_pixel.zero_grad()
            loss_dis_pixel.backward()
            optimizer_D_pixel.step()






            batch_done = epoch * len(dataloader) + i
            print(
                '[epoch: %d/%d] [batch: %d/%d] [D loss: %f pixel loss %f] [generator loss: %f ] [finished: %f percent]'
                % (
                    epoch,
                    config.n_epochs,
                    i,
                    len(dataloader),
                    loss_dis.item(),
                    loss_dis_pixel.item(),
                    # loss_generator.item(),

                    loss_gan.item(),
                    # loss_pixel.item(),
                    (batch_done) / (config.n_epochs * len(dataloader)) * 100

                )
            )
            if batch_done % config.sample_interval == 0:
                with torch.no_grad():
                    test_image(batch_done, dataloader_val, stain_generator,config.loss_type)
            torch.cuda.empty_cache()
        # if (epoch+1)%config.alpha_decay==0:
        #     stain_generator.alpha=stain_generator.alpha-1/config.n_epochs
        lr_scheduler_G.step()
        # lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


        if (epoch+1) % config.checkpoints_interval == 0:
            torch.save(stain_generator.state_dict(),
                       'saved_models_T/%s/generator_stained%d.pth' % (config.loss_type,epoch))
            # torch.save(discriminator.state_dict(),
            #            'saved_models/%s/discriminator%d.pth' % (config.loss_type,epoch))
            torch.save(pixel_discriminator.state_dict(),
                       'saved_models_T/%s/discriminator_unstained%d.pth' % (config.loss_type,epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_images', type=str, default='images/unstained', help='save image generate')
    parser.add_argument('--path', type=str, default='finaldataset', help='dataset path')
    parser.add_argument('--mode', type=str, default='all_heart_data', help='dataset mode')
    parser.add_argument('--train_mode', type=str, default='train', help='train mode')
    parser.add_argument('--ngf', type=int, default=128, help='num channel filtes')
    parser.add_argument('--num_channel', type=int, default=3, help='num channels')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    parser.add_argument('--n_epochs', type=int, default=200, help='train epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=0, help='epoch')
    parser.add_argument('--decay_epoch', type=int, default=100, help='decay epoch')
    parser.add_argument('--ngpu', type=int, default=1, help='ngpu')
    parser.add_argument('--sample_interval', type=int, default=200, help='batch size')
    parser.add_argument('--input_shape', type=tuple, default=(3, 256, 256), help='input shape')
    parser.add_argument('--checkpoints_interval', type=int, default=20, help='check')
    parser.add_argument('--lambda_gp', type=int, default=10, help='check')
    parser.add_argument('--lambda_cyc', type=int, default=10, help='check')
    parser.add_argument('--lambda_identity', type=int, default=10, help='check')
    parser.add_argument('--num_residual_blocks', type=int, default=9, help='check')
    parser.add_argument('--save_checkpoint', type=str, default='saved_models/unstained', help='check')
    parser.add_argument('--loss_type', type=str, default='LS-loss', help='WGAN-GP,BCE-loss,LS-loss,Logistic-loss')
    parser.add_argument('--use_fm', type=bool, default=True, help='fm loss')
    parser.add_argument('--backbone_name', type=str, default='resnet50', help='')
    parser.add_argument('--num_channels', type=int, default=3, help='')
    parser.add_argument('--ndf', type=int, default=512, help='')
    parser.add_argument('--fndf', type=int, default=128, help='')
    parser.add_argument('--pndf', type=int, default=64, help='')
    parser.add_argument('--alpha_decay', type=int, default=10, help='')
    # parser.add_argument('--sample_images', type=str, default='images', help='check')

    config = parser.parse_args()
    print(config)
    main(config)
