import imageio
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import cv2

import numpy as np
import pdb, os, argparse
from datetime import datetime
from model.ResNet_models import Generator
from data import get_loader
from utils import adjust_lr
from scipy import misc
from utils import l2_regularisation
import smoothness
from sklearn.metrics import precision_score, recall_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = '1'    

TRAIN = False
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--sm_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--reg_weight', type=float, default=0.001, help='weight for regularization term')
parser.add_argument('--lat_weight', type=float, default=5.0, help='weight for latent loss')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
parser.add_argument('--depth_loss_weight', type=float, default=1, help='weight for depth loss')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()  # Minimize dice loss

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).sum()




def compute_metrics(pred, gt, threshold=0.3):
    """
    Computes Precision, Recall, and Dice Score for segmentation.
    Args:
        pred (torch.Tensor): Predicted mask (logits), shape (B, 1, H, W)
        gt (torch.Tensor): Ground truth mask (binary), shape (B, 1, H, W)
        threshold (float): IoU threshold to count as True Positive.

    Returns:
        precision (float), recall (float), dice (float)
    """
    # Convert predictions to binary mask
    pred_bin = (torch.sigmoid(pred) > 0.2).float()
    target_bin = (gt > 0.5).float()

    intersection = (pred_bin * target_bin).sum(dim=(2, 3), keepdim=True)  # Keep dimensions
    union = (pred_bin + target_bin).sum(dim=(2, 3), keepdim=True) - intersection
    union = union.clamp(min=1e-6)
    iou = intersection / union
    tp = (iou > threshold).float().sum()
    fp = (pred_bin.sum(dim=(2, 3)) > 0).float().sum() - tp
    fn = (target_bin.sum(dim=(2, 3)) > 0).float().sum() - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    dice = (2 * intersection) / (pred_bin.sum(dim=(2, 3), keepdim=True) + target_bin.sum(dim=(2, 3), keepdim=True) + 1e-6)
    dice = dice.mean().item()

    return precision.item(), recall.item(), dice
## visualize predictions and gt
def visualize_uncertainty_post_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_post_int.png'.format(kk)
        imageio.imwrite(save_path + name, (pred_edge_kk * 255).astype(np.uint8))

def visualize_uncertainty_prior_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_prior_int.png'.format(kk)
        imageio.imwrite(save_path + name, (pred_edge_kk * 255).astype(np.uint8))

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        imageio.imwrite(save_path + name, (pred_edge_kk * 255).astype(np.uint8))

## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def focal_loss(pred, target, alpha=0.25, gamma=2.0):

    pred_prob = torch.sigmoid(pred)
    pt = pred_prob * target + (1 - pred_prob) * (1 - target)  # p_t for each pixel
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-6)
    return loss.mean()


def overlay_masks_gray(image_rgb, gt_mask, pred_mask, alpha=0.5):


    gray = np.dot(image_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    # Make a 3-channel grayscale image.
    gray_img = np.stack([gray, gray, gray], axis=-1)

    # Prepare background as float for blending.
    overlay = gray_img.astype(np.float32)
    background = gray_img.astype(np.float32)

    # Define colors in RGB.
    yellow = np.array([255, 255, 0], dtype=np.float32)  # Ground truth only
    blue = np.array([0, 0, 255], dtype=np.float32)  # Prediction only
    green = np.array([0, 255, 0], dtype=np.float32)  # Overlap

    # Create boolean masks.
    gt_bin = (gt_mask > 0)
    pred_bin = (pred_mask > 0)

    # Conditions:
    # Pixels where both masks are active.
    both = gt_bin & pred_bin
    # Pixels with ground truth only.
    only_gt = gt_bin & ~pred_bin
    # Pixels with prediction only.
    only_pred = pred_bin & ~gt_bin

    # Blend for pixels where both are active.
    overlay[both] = background[both] * (1 - alpha) + green * alpha
    # Blend for pixels where only ground truth is active.
    overlay[only_gt] = background[only_gt] * (1 - alpha) + yellow * alpha
    # Blend for pixels where only prediction is active.
    overlay[only_pred] = background[only_pred] * (1 - alpha) + blue * alpha

    # Clip the results and convert to uint8.
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
generator.load_state_dict(torch.load('./models/Model_100_gen.pth', map_location=lambda storage, loc: storage.cuda(0)))
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=[opt.beta1_gen, 0.999],weight_decay=0.01)

for seed in range(1,2):
    print('Seed: {}'.format(seed))
    ## load data
    image_root = f'./placenta_data/seed{seed}/train/img/'
    gt_root = f'./placenta_data/seed{seed}/train/gt/'
    depth_root = f'./placenta_data/seed{seed}/train/depth/'
    gray_root = f'./placenta_data/seed{seed}/train/gray/'
    train_loader, training_set_size = get_loader(image_root, gt_root, depth_root, gray_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    train_z = torch.FloatTensor(training_set_size, opt.latent_dim).normal_(0, 1).cuda()
    val_image_root = f'./placenta_data/seed{seed}/val/img/'
    val_gt_root = f'./placenta_data/seed{seed}/val/gt/'
    val_depth_root = f'./placenta_data/seed{seed}/val/depth/'
    val_gray_root = f'./placenta_data/seed{seed}/val/gray/'

    val_loader, val_set_size = get_loader(val_image_root, val_gt_root, val_depth_root, val_gray_root,
                                          batchsize=opt.batchsize, trainsize=opt.trainsize)


    test_image_root = f'./placenta_data/seed{seed}/test/img/'
    test_gt_root = f'./placenta_data/seed{seed}/test/gt/'
    test_depth_root = f'./placenta_data/seed{seed}/test/depth/'
    test_gray_root = f'./placenta_data/seed{seed}/test/gray/'

    test_loader, test_set_size = get_loader(test_image_root, test_gt_root, test_depth_root, test_gray_root,
                                          batchsize=opt.batchsize, trainsize=opt.trainsize)


    ## define loss

    mse_loss = torch.nn.MSELoss(reduction='mean')  # Mean reduction for MSE Loss
    CE = torch.nn.BCELoss(reduction='mean')  # Mean reduction for Binary Cross Entropy Loss
    smooth_loss = smoothness.smoothness_loss(size_average=True)
    # --- Grid search combination: lambda_focal=1, lambda_dice=1, reg_weight=0.001, lat_weight=5, vae_loss_weight=0.6, depth_loss_weight=0.1 ---


    print("Let's Play!")
    # /Users/michlis/PycharmProjects/placenta_detection/ucnet/grid_search_results/focal_1_dice_1_reg_0.001_lat_5_vae_0.6_depth_1/best_model.pth
    best_model_path = "grid_search_results/focal_1_dice_1_reg_0.001_lat_5_vae_0.4_depth_1/best_model.pth"
    if TRAIN:
        patience = opt.patience
        lambda_focal = 1
        lambda_dice = 1
        epochs_no_improve = 0
        best_val_loss = float("inf")
        best_val_dice = -float("inf")
        for epoch in range(1, opt.epoch + 1):
            print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

            ### Training Phase
            generator.train()
            train_loss = 0.0
            train_precision = 0.0
            train_recall = 0.0
            train_dice = 0.0
            for i, pack in enumerate(train_loader, start=1):
                images, gts, depths, grays, index_batch = pack
                images, gts, depths, grays = images.to(device), gts.to(device), depths.to(device), grays.to(device)

                pred_post, pred_prior, latent_loss, depth_pred_post, depth_pred_prior = generator(images, depths, gts)



                ## Compute losses
                reg_loss = l2_regularisation(generator.xy_encoder) + \
                           l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
                smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), gts)
                reg_loss = opt.reg_weight * reg_loss

                dice_loss_comp = dice_loss(pred_post, gts)

                depth_loss_post = opt.depth_loss_weight * mse_loss(torch.sigmoid(depth_pred_post), depths)
                sal_loss = structure_loss(pred_post, gts) + smoothLoss_post + depth_loss_post
                anneal_reg = min(1.0, epoch / opt.epoch)  # Linear annealing
                latent_loss = opt.lat_weight * anneal_reg * latent_loss
                gen_loss_cvae = opt.vae_loss_weight * (sal_loss + latent_loss)

                smoothLoss_prior = opt.sm_weight * smooth_loss(torch.sigmoid(pred_prior), gts)
                depth_loss_prior = opt.depth_loss_weight * mse_loss(torch.sigmoid(depth_pred_prior), depths)
                gen_loss_gsnn = (1 - opt.vae_loss_weight) * (
                            structure_loss(pred_prior, gts) + smoothLoss_prior + depth_loss_prior)
                focal_loss_val = focal_loss(pred_post, gts)

                gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss + lambda_dice * dice_loss_comp + lambda_focal * focal_loss_val

                generator_optimizer.zero_grad()
                gen_loss.backward()
                generator_optimizer.step()

                train_loss += gen_loss.item()
                precision, recall, dice = compute_metrics(pred_post.detach(), gts.detach())

                train_precision += precision
                train_recall += recall
                train_dice += dice

            # Normalize metrics
            train_loss /= len(train_loader)
            train_precision /= len(train_loader)
            train_recall /= len(train_loader)
            train_dice /= len(train_loader)
            ### Validation Phase



            generator.eval()
            val_loss, val_precision, val_recall, val_dice = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                for images, gts, depths, grays, _ in val_loader:
                    images, gts, depths, grays = images.to(device), gts.to(device), depths.to(device), grays.to(device)

                    pred_post, pred_prior, latent_loss, depth_pred_post, depth_pred_prior = generator(images, depths, gts)

                    # Compute validation loss
                    reg_loss = l2_regularisation(generator.xy_encoder) + \
                               l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
                    smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), gts)
                    depth_loss_post = opt.depth_loss_weight * mse_loss(torch.sigmoid(depth_pred_post), depths)
                    sal_loss = structure_loss(pred_post, gts) + smoothLoss_post + depth_loss_post
                    latent_loss = opt.lat_weight * anneal_reg * latent_loss
                    gen_loss_cvae = opt.vae_loss_weight * (sal_loss + latent_loss)

                    smoothLoss_prior = opt.sm_weight * smooth_loss(torch.sigmoid(pred_prior), gts)
                    depth_loss_prior = opt.depth_loss_weight * mse_loss(torch.sigmoid(depth_pred_prior), depths)
                    gen_loss_gsnn = (1 - opt.vae_loss_weight) * (
                            structure_loss(pred_prior, gts) + smoothLoss_prior + depth_loss_prior)
                    dice_loss_comp = dice_loss(pred_post, gts)
                    focal_loss_val = focal_loss(pred_post, gts)

                    gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss + lambda_dice * dice_loss_comp + lambda_focal * focal_loss_val
                    val_loss += gen_loss.item()

                    # Compute Precision, Recall, and Dice
                    precision, recall, dice = compute_metrics(pred_post, gts)
                    val_precision += precision
                    val_recall += recall
                    val_dice += dice

            # Normalize metrics
            val_loss /= len(val_loader)
            val_precision /= len(val_loader)
            val_recall /= len(val_loader)
            val_dice /= len(val_loader)

            print(
                f"Epoch {epoch}/{opt.epoch} | Val Loss: {val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | Dice: {val_dice:.4f}")
            print(
                f"Epoch {epoch}/{opt.epoch} | Train Loss: {train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | Dice: {train_dice:.4f}")

            # Early Stopping & Model Saving
            if val_loss < best_val_loss:
                print("✅ New best model found! Saving...")
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(generator.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping after {epoch} epochs!")
                break

            adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)


    generator.eval()
    generator.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage.cuda(0)))
    generator.eval()
    test_loss, test_precision, test_recall, test_dice = 0.0, 0.0, 0.0, 0.0
    best_test_dice = 0.0
    with torch.no_grad():
        for images, gts, depths, grays, _ in test_loader:
            images, gts, depths, grays = images.to(device), gts.to(device), depths.to(device), grays.to(device)

            pred_post, pred_prior, latent_loss, depth_pred_post, depth_pred_prior = generator(images, depths, gts)

            precision, recall, dice = compute_metrics(pred_post, gts)
            test_precision += precision
            test_recall += recall
            test_dice += dice

    # Normalize metrics
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_dice /= len(test_loader)

    print(f"\nFinal Test results| Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | Dice: {test_dice:.4f}")

    with torch.no_grad():
        for batch_idx, (images, gts, depths, grays, _) in enumerate(test_loader):
            images = images.to(device)
            gts = gts.to(device)
            depths = depths.to(device)
            grays = grays.to(device)

            # Get model predictions (using pred_post for example)
            pred_post, pred_prior, latent_loss, depth_pred_post, depth_pred_prior = generator(images, depths, gts)

            for i in range(images.size(0)):
                # Convert image tensor (assumed to be RGB in [0,1]) to a NumPy array in [0,255].
                image_np = images[i].detach().cpu().numpy().transpose(1, 2, 0)
                image_np = (image_np * 255).astype(np.uint8)  # still in RGB

                # Convert ground truth mask to binary NumPy array.
                gt_np = gts[i].detach().cpu().numpy().squeeze()
                gt_bin = (gt_np > 0.5).astype(np.uint8)

                # Convert predicted mask to binary NumPy array.
                pred_prob = torch.sigmoid(pred_post[i]).detach().cpu().numpy().squeeze()
                pred_bin = (pred_prob > 0.5).astype(np.uint8)

                # Create the overlay image (the function expects an RGB image and returns an RGB image).
                overlay_image_rgb = overlay_masks_gray(image_np, gt_bin, pred_bin, alpha=0.5)

                # Convert the RGB overlay to BGR for saving with OpenCV.
                overlay_image_bgr = cv2.cvtColor(overlay_image_rgb, cv2.COLOR_RGB2BGR)

                # Save the image using OpenCV.
                save_name = f"./temp/overlay_batch{batch_idx}_img{i}.png"
                cv2.imwrite(save_name, overlay_image_bgr)

# Final Test Loss: 1577.4854 | Precision: 0.8782 | Recall: 0.8355 | Dice: 0.4986

# Final Test Loss: 1156.4614 | Precision: 0.9509 | Recall: 0.9169 | Dice: 0.5660


#
# Final Test Loss: 1168.8141 | Precision: 0.9086 | Recall: 0.8859 | Dice: 0.5462
#
# Final Test Loss: 1158.7335 | Precision: 0.9852 | Recall: 0.9659 | Dice: 0.6627
# Seed: 3
#
# Final Test Loss: 1159.6449 | Precision: 0.9492 | Recall: 0.9420 | Dice: 0.6331
# Seed: 4
#
# Final Test Loss: 1158.9216 | Precision: 0.9752 | Recall: 0.9544 | Dice: 0.6977


# diffrent time split
# Final Test results| Precision: 0.9268 | Recall: 0.7652 | Dice: 0.4263-
#dice loss = 1000
# Final Test results| Precision: 0.8217 | Recall: 0.8733 | Dice: 0.4832


#BEST MODEL - Final Test results| Precision: 0.9100 | Recall: 0.8920 | Dice: 0.5064
#best_model_path = "grid_search_results/focal_1_dice_1_reg_0.001_lat_5_vae_0.4_depth_1/best_model.pth"
