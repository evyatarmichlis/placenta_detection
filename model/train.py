import os
import torch
import random
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid

import consts
from Code.lib.model_pvt import XMSNet
from Code.utils.data import get_loader, test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from Code.utils.options import opt
import torch.nn as nn

# set the device for training

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU {}'.format(opt.gpu_id))
# print('USE GPU {}')

cudnn.benchmark = True

# build the model
model = XMSNet(pvt_pretrained=True)
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


def set_random_seed(seed):
    seed = int(seed)  # """Set random seed for reproduce"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# set the path

set_random_seed(3407)
train_image_root = opt.rgb_label_root
train_gt_root = opt.gt_label_root
train_depth_root =  opt.depth_label_root

val_image_root =  opt.val_rgb_root
val_gt_root =  opt.val_gt_root
val_depth_root =   opt.val_depth_root
save_path = (r"/cs/usr/evyatar613/josko_lab"
             r"/Checkpoint/MRNet/")

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root, train_depth_root, batchsize=opt.batchsize,
                          trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root, val_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("MRNet-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
loss_kl = nn.KLDivLoss(reduction='batchmean')
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0

print(len(train_loader))


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weights = torch.where(mask==0,torch.tensor(2.0),torch.tensor(1.0)) #TODO check if works! I ADDED THIS
    wbce = F.binary_cross_entropy_with_logits(pred, mask, weight=weights,reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_coefficient(pred, mask):
    pred_binary = (pred > 0.5).float()
    mask_binary = (mask > 0.5).float()

    intersection = torch.sum(pred_binary * mask_binary)
    total = torch.sum(pred_binary) + torch.sum(mask_binary)

    dice = (2. * intersection + 1e-7) / (total + 1e-7)

    return dice


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

            ##
            S0, S1, S2, S3, S4, S5, S6 = model(images, depths)

            loss0 = 0.2 * structure_loss(S0, gts)
            loss1 = 0.4 * structure_loss(S1, gts)
            loss2 = 0.6 * structure_loss(S2, gts)
            loss3 = 0.8 * structure_loss(S3, gts)
            loss4 = 0.3 * structure_loss(S4, gts)
            loss5 = 0.6 * structure_loss(S5, gts)
            loss6 = 0.9 * structure_loss(S6, gts)

            losskl1 = loss_kl(F.log_softmax(S4, dim=1), F.softmax(S5, dim=1)) + loss_kl(F.log_softmax(S5, dim=1),
                                                                                        F.softmax(S4, dim=1))
            losskl2 = loss_kl(F.log_softmax(S5, dim=1), F.softmax(S6, dim=1)) + loss_kl(F.log_softmax(S6, dim=1),
                                                                                        F.softmax(S5, dim=1))
            loss_seg = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.5 * losskl1 + 0.5 * losskl2

            loss = loss_seg
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 50 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                             format(epoch, opt.epoch, i, total_step, loss1.data))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            S0, S1, S2, S3, S4, S5, S6 = model(image, depth)
            res = S6
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'XMSNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")

    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # train
        train(train_loader, model, optimizer, epoch, save_path)

        # test
        val(test_loader, model, epoch, save_path)
