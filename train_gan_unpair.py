import os
import torch.optim
import numpy as np
import argparse

from torch.utils import data

from ssdn.network import NoiseNetwork
from lossfunction_dual import *
from datasets.DenoisingDatasets_unpair import BenchmarkTrain
from datasets.DenoisingDatasets import SIDD_VAL
import torch.nn as nn
import time
from lossfunction import CharbonnierLoss
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# import cv2
from skimage.metrics import peak_signal_noise_ratio
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import matplotlib.pyplot as plt
def save_img(im_noisy,noise_red,noise_blue,noise,noise_full2,global_step):
    im_noisy = np.int8(np.transpose(im_noisy.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    noise_red = np.int8(np.transpose(noise_red.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    noise_blue = np.int8(np.transpose(noise_blue.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    noise = np.int8(np.transpose(noise.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    noise_full2 = np.int8(np.transpose(noise_full2.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    # denoise_red = (denoise_red - np.min(denoise_red)) / (np.max(denoise_red) - np.min(denoise_red))


    fig, axs = plt.subplots(3,3,figsize=(20,20))
    axs[0,0].imshow(im_noisy,cmap='gray')
    axs[0,0].set_title("im_noisy")
    axs[0,2].imshow(noise_red,cmap='gray')
    axs[0,2].set_title("noise_red")
    axs[1,0].imshow(noise_blue,cmap='gray')
    axs[1,0].set_title("noise_blue")
    axs[2,0].imshow(noise,cmap='gray')
    axs[2,0].set_title("noise")
    axs[2,1].imshow(noise_full2,cmap='jet')
    axs[2,1].set_title("noise_full2")

    plt.axis("off")
    fig.suptitle(str(global_step)+ " : ",fontsize=50)

    # plt.imshow(im_noisy,cmap='gray')
    # plt.show()
    folder = "img_gan_unpair_dynamic_zero/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.savefig(folder+str(global_step)+".png")
    # plt.imsave(im_noisy,str(global_step) +".png")

def train(config):
    # Train in CPU
    if config.gpu_id == -1:
        device = torch.device('cpu')
    # Train in GPU
    else:
        device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    batch_size = config.batch_size
    epochs = config.epochs
    data_dir = config.data_dir
    num_workers = config.num_workers
    N = config.N

    # Load dataset
    train_dataset = BenchmarkTrain(data_dir, N * batch_size, pch_size=256)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    # val_dataset = SIDD_VAL('../validation/RAW/')
    val_dataset = SIDD_VAL('/vinai/tampm2/SIDD')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    lossfunc = BasicLoss().cuda()
    # lossfunc_fake = BasicLoss_fake().cuda()
    # loss_diss = nn.BCELoss()
    # loss_fourier = FourierLoss().cuda()
    # lossMSE = nn.MSELoss(reduction='mean')
    lossCha = CharbonnierLoss()
    # Load model
    net = NoiseNetwork(out_channels=4).to(device)
    # model_dis = DiscriminatorLinear(in_chn=4).to(device)

    # Optimization
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # optimizer_dis = torch.optim.Adam(
    #     model_dis.parameters(),
    #     lr=config.lr*0.2,weight_decay=config.weight_decay
    # )
    milestones = [10, 20, 25, 30, 35, 40, 45, 50]
    # milestones = [3, 6, 10, 15,20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.5)
    # scheduler_dis = torch.optim.lr_scheduler.MultiStepLR(optimizer_dis, milestones, 0.5)

    cur_epoch = 0
    global_step = 0
    if config.pretrain_model:
        print('Loading pretrained model.')
        checkpoint = torch.load(config.pretrain_model)
        net.load_state_dict(checkpoint['model_state_dict'])
        # model_dis.load_state_dict(checkpoint['model_dis_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer_dis.load_state_dict(checkpoint['optimizer_dis_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # scheduler_dis.load_state_dict(checkpoint['scheduler_dis_state_dict'])
        cur_epoch = checkpoint['cur_epoch']
        global_step = checkpoint['global_step']
        print('Loaded pretrained model sucessfully.')

    with torch.no_grad():
        psnr = []
        for ii, data in enumerate(val_loader):
            im_noisy = data[0].to(device)
            im_gt = data[1].to(device)
            # print(im_gt.size())

            out = net(im_noisy)
            noise = torch.clamp(out, -1, 1)
            restored_2 = torch.clamp(im_noisy - noise, 0, 1)
            restored_2 = np.transpose(restored_2.cpu().numpy(), [0, 2, 3, 1])
            im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])
            # print(im_gt.shape)
            psnr.extend([peak_signal_noise_ratio(restored_2[i], im_gt[i], data_range=1) for i in range(im_gt.shape[0])])
        print('psnr={:.4e}'.format(np.mean(psnr)))

    for epoch in range(cur_epoch, epochs):
        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        loss_per_epoch = {x: [] for x in ['loss',"dis"]}
        lr = scheduler.get_last_lr()[0]
        tic = time.time()
        for ii, data in enumerate(train_loader):
            global_step += 1
            im_noisy = data[0].to(device)
            im_gt = data[1].to(device)

            input_red = data[2].to(device)
            input_blue = data[3].to(device)
            im_gt2 = data[7].to(device)

            for _ in range(1):
                net.train()
                # Inference
                output_red = net(input_red)
                output_blue = net(input_blue)
                # with torch.no_grad():
                noise_full = net(im_noisy)
                # print(noise_full)

                # noise_red = torch.clamp(output_red, -1, 1)
                noise_red = output_red
                # noise_blue = torch.clamp(output_blue, -1, 1)
                noise_blue = output_blue
                # im_restore_noise = torch.clamp(noise_full, -1, 1)
                im_restore_noise = noise_full

                denoise_red = input_red - noise_red
                denoise_blue = input_blue - noise_blue
                im_restore = im_noisy - im_restore_noise
                # print(im_gt2.size())
                # print(im_gt.size())
                # print(im_restore_noise.size())
                noise_clean2 = net(im_gt2)
                """
                img_noise2 = im_gt2 + im_restore_noise
                noise_full2 = net(img_noise2)
                # print(noise_full2)
                # noise_full2 = torch.clamp(noise_full2, -1, 1)
                im_full2 = img_noise2 - noise_full2
                """
                # loss = lossfunc(im_noisy,im_restore,im_restore_noise,input_red, denoise_red, noise_red, input_blue, denoise_blue, noise_blue)+ 0.0005*loss_
                loss = lossfunc(im_noisy,im_restore,im_restore_noise,input_red, denoise_red, noise_red, input_blue, denoise_blue, noise_blue)
                # loss += min(1.0,float(epoch)*1.0/20.0)*lossCha(im_full2,im_gt2)
                # loss += min(0.1,float(epoch)*1.0/50.0)*lossCha(im_full2,im_gt2)     # 42  img_gan_unpair_dynamic_char
                loss += min(1.0, float(epoch) * 1.0 / 50.0) * lossCha(noise_clean2, torch.zeros_like(noise_clean2))     # img_gan_unpair_dynamic_zero
                optimizer.zero_grad()
                loss.backward()
                # save_img(im_noisy, im_gt, output_red, im_restore, mse_img, r1, r2, global_step)
                optimizer.step()
            loss_per_epoch['loss'].append(loss.item())
            # loss_per_epoch['dis'].append(loss_.item())
            # loss_per_epoch['dis'].append(0)

            # --------------------------------------------------------------
            # Log
            # --------------------------------------------------------------
            # save_img(im_noisy, im_gt, denoise_red, noise_red, denoise_blue, noise_blue, global_step)

            if (ii + 1) % config.print_freq == 0:
                mean_loss = np.mean(loss_per_epoch['loss'])
                mean_dis = np.mean(loss_per_epoch['dis'])
                with torch.no_grad():
                    out = net(im_noisy)
                    noise = torch.clamp(out,-1,1)
                    restored_2 = torch.clamp(im_noisy - noise,0,1)
                    restored_2 = np.transpose(restored_2.cpu().numpy(), [0, 2, 3, 1])
                    im_gt_np = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])
                    im_noisy_np = np.transpose(im_noisy.cpu().numpy(), [0, 2, 3, 1])

                    psnr_ori = np.mean(
                        [peak_signal_noise_ratio(im_noisy_np[i], im_gt_np[i], data_range=1) for i in range(im_gt_np.shape[0])])
                    psnr2 = np.mean(
                        [peak_signal_noise_ratio(restored_2[i], im_gt_np[i], data_range=1) for i in range(im_gt_np.shape[0])])
                    # save_img(im_noisy,noise_red,noise_blue,noise,noise_full2,global_step)

                log_str = '[Epoch:{:>2d}/{:<2d}] : {:0>4d}/{:0>4d}, Loss={:.1e},Loss_dis={:.1e}, psnr_ori = {:.2e} , pnsr2={:.2e}, lr={:.1e}'
                print(log_str.format(epoch + 1, epochs, ii + 1, N, mean_loss,mean_dis, psnr_ori , psnr2, lr))
                with torch.no_grad():
                    psnr = []
                    psnr2 = []
                    for ii, data in enumerate(val_loader):
                        im_noisy = data[0].to(device)
                        im_gt = data[1].to(device)
                        out = net(im_noisy)
                        noise = torch.clamp(out, -1, 1)
                        restored_2 = torch.clamp(im_noisy - noise,0,1)
                        restored_2 = np.transpose(restored_2.cpu().numpy(), [0, 2, 3, 1])
                        im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

                        psnr2.extend(
                            [peak_signal_noise_ratio(restored_2[i], im_gt[i], data_range=1) for i in range(im_gt.shape[0])])
                    print(' ,  psnr2={:.4e}  '.format(np.mean(psnr2)))
        # --------------------------------------------------------------
        ###############################################################
        # --------------------------------------------------------------
        # Save after some loops
        # --------------------------------------------------------------
        scheduler.step()
        # scheduler_dis.step()
        if (epoch + 1) % config.save_model_freq == 0 or epoch + 1 == epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(config.model_dir, model_prefix + str(epoch + 1) + '.pth')
            torch.save({
                'model_state_dict': net.state_dict(),
                # 'model_dis_state_dict': model_dis.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'optimizer_dis_state_dict': optimizer_dis.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'scheduler_dis_state_dict': scheduler_dis.state_dict(),
                'global_step': global_step,
                'cur_epoch': epoch + 1},
                save_path_model)
        toc = time.time()

        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                psnr = []
                psnr2 = []
                for ii, data in enumerate(val_loader):
                    im_noisy = data[0].to(device)
                    im_gt = data[1].to(device)
                    out = net(im_noisy)
                    noise = torch.clamp(out, -1, 1)
                    restored_2 = torch.clamp(im_noisy - noise, 0, 1)
                    restored_2 = np.transpose(restored_2.cpu().numpy(), [0, 2, 3, 1])
                    im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

                    psnr2.extend(
                        [peak_signal_noise_ratio(restored_2[i], im_gt[i], data_range=1) for i in range(im_gt.shape[0])])
                print('psnr={:.4e}  ,  psnr2={:.4e}  '.format(np.mean(psnr),np.mean(psnr2)))
        print('This epoch take time {:.2f}'.format(toc - tic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument('--data_dir', type=str,
                        default="../crop_medium")

    parser.add_argument('--num_workers', type=int, default=16)
    # parser.add_argument('--pretrain_model', type=str, default='img_dual_345_gan_denoise_fake/model_18.pth')
    parser.add_argument('--pretrain_model', type=str, default=None)

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=1000)

    parser.add_argument('--N', type=int, default=2500)

    parser.add_argument('--model_dir', type=str, default="img_gan_unpair2/")

    config = parser.parse_args()
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    train(config)
