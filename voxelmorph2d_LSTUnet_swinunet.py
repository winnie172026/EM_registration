import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import statistics as stats

# from sharp_swin_transformer_unet import swinUNet
from swin_transformer_unet_o import swinUNet

torch.manual_seed(42)
torch.cuda.manual_seed(42)
use_gpu = torch.cuda.is_available()

# class UNet(nn.Module):
#     def contracting_block(self, in_channels, out_channels, kernel_size=3):
#         """
#         This function creates one contracting block
#         """
#         block = torch.nn.Sequential(
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#         )
#         return block

#     def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This function creates one expansive block
#         """
#         block = torch.nn.Sequential(
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
#             torch.nn.BatchNorm2d(mid_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
#             torch.nn.BatchNorm2d(mid_channel),
#             torch.nn.ReLU(),
#             torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#         )
#         return block

#     def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This returns final block
#         """
#         block = torch.nn.Sequential(
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
#                     torch.nn.BatchNorm2d(mid_channel),
#                     torch.nn.ReLU(),
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
#                     torch.nn.BatchNorm2d(out_channels),
#                     torch.nn.ReLU()
#                 )
#         return block

#     def __init__(self, in_channel, out_channel):
#         super(UNet, self).__init__()
#         #Encode
#         self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
#         self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode2 = self.contracting_block(32, 64)
#         self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode3 = self.contracting_block(64, 128)
#         self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
#         # Bottleneck
#         mid_channel = 128
#         self.bottleneck = torch.nn.Sequential(
#                                 torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
#                                 torch.nn.BatchNorm2d(mid_channel * 2),
#                                 torch.nn.ReLU(),
#                                 torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
#                                 torch.nn.BatchNorm2d(mid_channel),
#                                 torch.nn.ReLU(),
#                                 torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
#                                 torch.nn.BatchNorm2d(mid_channel),
#                                 torch.nn.ReLU(),
#                             )
#         # Decode
#         self.conv_decode3 = self.expansive_block(256, 128, 64)
#         self.conv_decode2 = self.expansive_block(128, 64, 32)
#         self.final_layer = self.final_block(64, 32, out_channel)

#     def crop_and_concat(self, upsampled, bypass, crop=False):
#         """
#         This layer crop the layer from contraction block and concat it with expansive block vector
#         """
#         if crop:
#             c = (bypass.size()[2] - upsampled.size()[2]) // 2
#             bypass = F.pad(bypass, (-c, -c, -c, -c))
#         return torch.cat((upsampled, bypass), 1)

#     def forward(self, x):
#         # Encode
#         encode_block1 = self.conv_encode1(x)
#         encode_pool1 = self.conv_maxpool1(encode_block1)
#         encode_block2 = self.conv_encode2(encode_pool1)
#         encode_pool2 = self.conv_maxpool2(encode_block2)
#         encode_block3 = self.conv_encode3(encode_pool2)
#         encode_pool3 = self.conv_maxpool3(encode_block3)
#         # Bottleneck
#         bottleneck1 = self.bottleneck(encode_pool3)
#         # Decode
#         decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
#         cat_layer2 = self.conv_decode3(decode_block3)
#         decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
#         cat_layer1 = self.conv_decode2(decode_block2)
#         decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
#         final_layer = self.final_layer(decode_block1)
#         return  final_layer
        
class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))
        batch_size, height, width, channels = im.shape
        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape
        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)



class VoxelMorph2d(nn.Module):
    def __init__(self, in_channels=3, use_gpu=False):
        super(VoxelMorph2d, self).__init__()
        self.unet = swinUNet(in_channels, 2)
        self.swinunet = swinUNet(in_channels, 2)
        self.spatial_transform = SpatialTransformation(use_gpu)
        if use_gpu:
            self.unet = self.unet.cuda()
            self.swinunet = self.swinunet.cuda()
            self.spatial_transform = self.spatial_transform.cuda()

    def forward(self, moving_image, fixed_image, batch_mask_fixed, batch_mask_moving):
        x = torch.cat([moving_image, fixed_image], dim=3).permute(0,3,1,2)
        # print("input: ", x.size())
        # print("moving_image: ", moving_image.size())
        # print("fixed_image: ", fixed_image.size())
        
        # deformation_matrix = self.unet(x).permute(0,2,3,1)
        # registered_image = self.spatial_transform(moving_image, deformation_matrix)
        # mask_registered_image_unet = self.spatial_transform(batch_mask_moving, deformation_matrix)    ### mask_registeder_image 与 mask_fixed 计算损失
        # mask_registered_image_unet_np = mask_registered_image_unet.cpu().detach().numpy()
        
        
        deformation_matrix = self.swinunet(x).permute(0,2,3,1)
        # print("deformation_matrix: ", deformation_matrix.size())
        registered_image = self.spatial_transform(moving_image, deformation_matrix)
        mask_registered_image = self.spatial_transform(batch_mask_moving, deformation_matrix)    ### mask_registeder_image 与 mask_fixed 计算损失
        # print("mask_registered_image: ", mask_registered_image.size())
        mask_registered_image_np = mask_registered_image.cpu().detach().numpy()

        # deformation_matrix1 = self.swinunet(x).permute(0,2,3,1)
        # # print("deformation_matrix: ", deformation_matrix.size())
        # registered_image_final = self.spatial_transform(registered_image, deformation_matrix1)
        # mask_registered_image_final = self.spatial_transform(mask_registered_image_unet, deformation_matrix1)    ### mask_registeder_image 与 mask_fixed 计算损失
        # # print("mask_registered_image: ", mask_registered_image.size())
        # mask_registered_image_final_np = mask_registered_image_final.cpu().detach().numpy()

 
        return registered_image, deformation_matrix, mask_registered_image
        # return registered_image_final, deformation_matrix1, mask_registered_image_final


def cross_correlation_loss(I, J, n):
    # print("coxelmorph2d I: ", I.size())
    # print("coxelmorph2d J: ", J.size())
    I = I.permute(0, 3, 1, 2)
    J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    if use_gpu:
        sum_filter = sum_filter.cuda()

    I_sum = F.conv2d(I, sum_filter, padding=1, stride=(1,1))
    J_sum = F.conv2d(J, sum_filter, padding=1, stride=(1,1))
    I2_sum = F.conv2d(I2, sum_filter, padding=1, stride=(1,1))
    J2_sum = F.conv2d(J2, sum_filter, padding=1, stride=(1,1))
    IJ_sum = F.conv2d(IJ, sum_filter, padding=1, stride=(1,1))
    win_size = n**2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
    return torch.mean(cc)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0

# Create a function
def pearson_loss(pred,true):
    
    
    true = true.flatten()
    pred = pred.flatten()
    
    # Create n, the number of observations in the data
    n = len(pred)
    
    # Create lists to store the standard scores
    standard_score_pred = []
    standard_score_true = []
    
    # Calculate the mean of x
    mean_pred = stats.mean(pred)
    
    # Calculate the standard deviation of x
    standard_deviation_pred = stats.stdev(pred)
    
    # Calculate the mean of y
    mean_true = stats.mean(true)
    
    # Calculate the standard deviation of y
    standard_deviation_true = stats.stdev(true)

    # For each observation in x
    for observation in pred: 
        
        # Calculate the standard score of x
        standard_score_pred.append((observation - mean_pred)/standard_deviation_pred) 

    # For each observation in y
    for observation in true:
        
        # Calculate the standard score of y
        standard_score_true.append((observation - mean_true)/standard_deviation_true)

    # Multiple the standard scores together, sum them, then divide by n-1, return that value
    pcc_loss = (sum([i*j for i,j in zip(standard_score_pred, standard_score_true)]))/(n-1)
    
    return pcc_loss

def vox_morph_loss(y, ytrue, n=9, lamda=0.01):
    cc = cross_correlation_loss(y, ytrue, n)
    # cc = pearson_loss(y, ytrue)
    # sm = smooothing_loss(y)
    sm = torch.mean((y - ytrue) ** 2)
    # print("CC Loss", cc, "Gradient Loss", sm)
    loss = -1.0 * cc + lamda * sm
    return loss

def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
        This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    # print("pred: ", pred)
    # print("target: ", target)
    pred = (pred>0.5).float()
    
    
    top = 2 *  torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice

def score_dice(y_pred, y_true, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    '''
    # skip the batch and class axis for calculating Dice score
    y_true = y_true.cpu().detach().numpy().astype(np.uint8)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.where(y_pred > 0.5, 1, 0)

    labels = np.concatenate([np.unique(a) for a in [y_true, y_pred]])
    labels = np.sort(np.unique(labels))

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(y_true == label, y_pred == label))
        bottom = np.sum(y_true == label) + np.sum(y_pred == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem.mean()


def mydice(pred, true):
    # pred = torch.from_numpy(pred)
    # true = torch.from_numpy(true)
    pred = (pred>0.5).float()

    intersection = (pred * true).sum()
    dice = (2. * intersection + 1) / (pred.sum() + true.sum() + 1)

    return dice


# def soft_dice(y_true, y_pred, epsilon=1e-6):
#     '''
#     Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
#     Assumes the `channels_last` format.
#     # Arguments
#         y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
#         y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
#         epsilon: Used for numerical stability to avoid divide by zero errors
#     '''
#     # skip the batch and class axis for calculating Dice score
#     y_true = y_true.cpu().detach().numpy().astype(np.uint8)
#     y_pred = torch.sigmoid(y_pred)
#     y_pred = y_pred.cpu().detach().numpy()

#     axes = tuple(range(1, len(y_pred.shape) - 1))
#     numerator = 2. * np.sum(y_pred * y_true, axes)
#     denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

#     dice = np.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch
#     return dice


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == i for i in range(num_classes)]
    mask = np.concatenate((_mask[0], _mask[1]), axis=-1)
    mask = np.array(mask).astype(np.uint8)
    return mask


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask