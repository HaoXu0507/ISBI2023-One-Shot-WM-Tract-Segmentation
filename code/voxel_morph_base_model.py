# cite the code from
# https://github.com/zuzhiang/VoxelMorph-torch/blob/master/Model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
import cv2 as cv
class U_Network(nn.Module):
    def __init__(self, dim,  enc_nf, dec_nf,out_channel=3, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2*9 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2*9, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], out_channel, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self, size):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)



    def forward(self, src, flow,mode='nearest'):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode)

def save_image(moving_img, fixed_img,moved_img, dir_name):
    fixed_img = fixed_img[0, 0, ...].cpu().detach().numpy()
    moving_img = moving_img[0, 0, ...].cpu().detach().numpy()
    moved_img = moved_img[0, 0, ...].cpu().detach().numpy()

    for i in [20,30,40,50,70,90]:
        fixed_img_i = np.rot90(fixed_img[:,:,i],-1)
        fixed= fixed_img_i if i == 20 else np.concatenate((fixed,fixed_img_i),axis=1)

        moving_img_i = np.rot90(moving_img[:, :, i],-1)
        moving = moving_img_i if i == 20 else np.concatenate((moving, moving_img_i), axis=1)

        moved_img_i = np.rot90(moved_img[:, :, i],-1)
        moved = moved_img_i if i == 20 else np.concatenate((moved, moved_img_i), axis=1)
    full= np.concatenate((np.concatenate((fixed,moved),axis=0),moving),axis=0)*255
    cv.imwrite(dir_name,full)

def save_image_intensity(moving_img, fixed_img,inversr_warp_img,moved_img, dir_name):
    fixed_img = fixed_img[0, 0, ...].cpu().detach().numpy()
    moving_img = moving_img[0, 0, ...].cpu().detach().numpy()
    moved_img = moved_img[0, 0, ...].cpu().detach().numpy()
    inversr_warp_img = inversr_warp_img[0, 0, ...].cpu().detach().numpy()
    for i in [20,30,40,50,70,90]:
        fixed_img_i = np.rot90(fixed_img[:,:,i],-1)
        fixed= fixed_img_i if i == 20 else np.concatenate((fixed,fixed_img_i),axis=1)

        inversr_warp_img_i = np.rot90(inversr_warp_img[:, :, i], -1)
        inversr_warp = inversr_warp_img_i if i == 20 else np.concatenate((inversr_warp, inversr_warp_img_i), axis=1)

        moving_img_i = np.rot90(moving_img[:, :, i],-1)
        moving = moving_img_i if i == 20 else np.concatenate((moving, moving_img_i), axis=1)

        moved_img_i = np.rot90(moved_img[:, :, i],-1)
        moved = moved_img_i if i == 20 else np.concatenate((moved, moved_img_i), axis=1)
    full= np.concatenate((np.concatenate((np.concatenate((fixed,moved),axis=0),inversr_warp),axis=0),moving),axis=0)*255
    cv.imwrite(dir_name,full)

def multilabel_intensity_transform(x,cha=9):
    '''
    #将72个channel的144*144*144加在一起，若超过1，则为1，否则为0.最后再复制到3通道
    Args:
        x: label [bs,72,144,144,144]

    Returns:
        mask_bs: [bs,3,144,144,144]
    '''
    [bs,c,l,h,w]=x.shape
    for i in range(bs):

        one = torch.ones_like(x[0,0,:,:,:])
        zero = torch.zeros_like(x[0,0,:,:,:])
        for j in range(c):
            if j ==0:
                coont=x[i,j,:,:,:]
            else:
                coont+=x[i,j,:,:,:]
        mask_one_channel= torch.where(coont>1,zero,one)
        if cha==3:
            mask = torch.stack((mask_one_channel,mask_one_channel,mask_one_channel),dim=0)
        elif cha==9:
            mask = torch.stack((mask_one_channel, mask_one_channel, mask_one_channel,mask_one_channel, mask_one_channel,mask_one_channel, mask_one_channel,mask_one_channel, mask_one_channel,), dim=0)
        mask=mask.unsqueeze(0)
        if i==0:
            mask_bs=mask
        else:
            mask_bs=torch.cat((mask_bs,mask),dim=0)
    return mask_bs

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    [1,9,144,144,144]
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    # sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    sum_filt = torch.ones([1, 9, *win]).cuda()

    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)