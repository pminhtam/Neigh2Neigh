import os
import torch
import torch.optim
import numpy as np
import argparse

from torch.utils import data

from ssdn.network import NoiseNetwork

from lossfunction import *
from datasets.DenoisingDatasets import BenchmarkTrain, SIDD_VAL
import torch.nn.functional as F
import torch.nn as nn
import time
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import matplotlib.pyplot as plt
def save_img(im_noisy,im_gt,output_red,im_restore,mse_img,r2,global_step,psnr):
    im_noisy = np.int8(np.transpose(im_noisy.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    im_gt = np.int8(np.transpose(im_gt.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    output_red = np.int8(np.transpose(output_red.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    im_restore = np.int8(np.transpose(im_restore.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    # mse_img = np.int8(np.transpose(mse_img.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]*255)
    full = (mse_img + r2)/2
    mse_img = np.transpose(mse_img.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]
    # mse_img = (mse_img - np.min(mse_img)) / (np.max(mse_img) - np.min(mse_img))
    # scale = 0.05
    scale = np.max(mse_img)
    mse_img = mse_img/scale
    mse_img = np.int8(mse_img*255)
    # mse_img = (mse_img-np.min(mse_img))/(np.max(mse_img)-np.min(mse_img))
    r2 = np.transpose(r2.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]
    r2 = r2 / scale
    r2 = np.int8(r2*255)

    full = np.transpose(full.cpu().detach().numpy(), [0, 2, 3, 1])[0,:,:,0]
    print(np.max(full))

    full = full / scale
    full = np.int8(full*255)
    # print(im_noisy)
    fig, axs = plt.subplots(3,2,figsize=(15,20))
    axs[0,0].imshow(im_noisy,cmap='gray')
    axs[0,0].set_title("im_noisy")
    axs[1,0].imshow(im_gt,cmap='gray')
    axs[1,0].set_title("im_gt")
    axs[2,0].imshow(im_restore,cmap='gray')
    axs[2,0].set_title("im_restore")
    axs[2,1].imshow(full,cmap='jet')
    axs[2,1].set_title("full loss")
    axs[0,1].imshow(mse_img,cmap='jet')
    axs[0,1].set_title("mse_img")
    axs[1,1].imshow(r2,cmap='jet')
    axs[1,1].set_title("red-blue")
    plt.axis("off")
    fig.suptitle(str(global_step)+ " : " + str(psnr),fontsize=50)

    # plt.imshow(im_noisy,cmap='gray')
    # plt.show()
    folder = "img_norm/"
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

    batch_size      = config.batch_size
    epochs          = config.epochs
    data_dir        = config.data_dir
    num_workers     = config.num_workers
    N               = config.N
    
    # Load dataset
    train_dataset = BenchmarkTrain(data_dir, N*batch_size, pch_size=256)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    # val_dataset = SIDD_VAL('../validation/RAW/')
    val_dataset = SIDD_VAL('/vinai/tampm2/SIDD')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    lossfunc = BasicLoss().cuda()
    # Load model
    net = NoiseNetwork().to(device)

    
    # Optimization
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # milestones = [10, 20, 25, 30, 35, 40, 45, 50]
    milestones = [20, 40, 60, 80, 100]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.5)


    cur_epoch = 0
    global_step = 0
    if config.pretrain_model:
        print('Loading pretrained model.')
        checkpoint = torch.load(config.pretrain_model)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['cur_epoch']
        global_step = checkpoint['global_step']
        print('Loaded pretrained model sucessfully.')

    with torch.no_grad():
        psnr = []
        for ii, data in enumerate(val_loader):
            im_noisy = data[0].to(device)
            im_gt = data[1].to(device)

            restored = torch.clamp(net(im_noisy), 0, 1)
            restored = np.transpose(restored.cpu().numpy(), [0, 2, 3, 1])
            im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

            psnr.extend([peak_signal_noise_ratio(restored[i], im_gt[i], data_range=1) for i in range(batch_size)])
        print('psnr={:.4e}'.format(np.mean(psnr)))
        
    for epoch in range(cur_epoch, epochs):
        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        loss_per_epoch = {x:[] for x in ['loss']}
        lr = scheduler.get_last_lr()[0]
        tic = time.time()
        for ii, data in enumerate(train_loader):
            net.train()
            global_step += 1
            im_noisy = data[0].to(device)
            im_gt = data[1].to(device)

            input_red = data[2].to(device)
            input_blue = data[3].to(device)

            mask_red = data[4].to(device)
            mask_blue = data[5].to(device)

            # Inference
            output_red = net(input_red)
            with torch.no_grad():
                im_restore = net(im_noisy)

            output_red = torch.clamp(output_red, 0, 1)
            im_restore = torch.clamp(im_restore, 0, 1)


            loss ,mse_img,r2 = lossfunc(output_red, input_blue, im_restore, mask_red, mask_blue)
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            # save_img(im_noisy, im_gt, output_red, im_restore, mse_img, r1, r2, global_step)

            optimizer.step()

            loss_per_epoch['loss'].append(loss.item())


            # --------------------------------------------------------------
            # Log
            # --------------------------------------------------------------
            if (ii+1) % config.print_freq == 0:
                mean_loss = np.mean(loss_per_epoch['loss'])
                with torch.no_grad():

                    restored = torch.clamp(net(im_noisy), 0, 1)
                    restored = np.transpose(restored.cpu().numpy(), [0, 2, 3, 1])
                    im_gt_np = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

                    psnr = np.mean([peak_signal_noise_ratio(restored[i], im_gt_np[i], data_range=1) for i in range(batch_size)])
                    save_img(im_noisy, im_gt, output_red, im_restore, mse_img, r2, global_step,psnr)

                log_str = '[Epoch:{:>2d}/{:<2d}] : {:0>4d}/{:0>4d}, Loss={:.1e}, pnsr={:.2e}, lr={:.1e}'
                print(log_str.format(epoch+1, epochs, ii+1, N, mean_loss, psnr, lr))

        # --------------------------------------------------------------
        ###############################################################
        # --------------------------------------------------------------
        # Save after some loops
        # --------------------------------------------------------------
        scheduler.step()
        if (epoch+1) % config.save_model_freq == 0 or epoch+1==epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(config.model_dir, model_prefix+str(epoch+1)+'.pth')
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'cur_epoch': epoch+1},
                save_path_model)

            model_state_prefix = 'model_state_'
            save_path_model_state = os.path.join(config.model_dir, model_state_prefix+str(epoch+1)+'.pth')
            torch.save(net.state_dict(), save_path_model_state)
        toc = time.time()

        if (epoch + 1)  % 1 == 0:
            with torch.no_grad():
                psnr = []
                for ii, data in enumerate(val_loader):
                    im_noisy = data[0].to(device)
                    im_gt = data[1].to(device)

                    restored = torch.clamp(net(im_noisy), 0, 1)
                    restored = np.transpose(restored.cpu().numpy(), [0, 2, 3, 1])
                    im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

                    psnr.extend([peak_signal_noise_ratio(restored[i], im_gt[i], data_range=1) for i in range(batch_size)])
                print('psnr={:.4e}'.format(np.mean(psnr)))
        print('This epoch take time {:.2f}'.format(toc-tic))
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument('--data_dir', type=str,
                        default="../crop_medium")


    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--pretrain_model', type=str, default=None)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=100)

    parser.add_argument('--N', type=int, default=2500)

    parser.add_argument('--model_dir', type=str, default="model")

    config = parser.parse_args()
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    train(config)
