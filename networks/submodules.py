import torch
from IPython.terminal.embed import embed
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-12

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def uncertainty_aware_samples(cur_depth, exp_var, ndepth, device, dtype, shape):
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) # (B, D, H, W)
    else:
        low_bound = -torch.min(cur_depth, exp_var)
        high_bound = exp_var
        assert ndepth > 1

        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []
        for i in range(int(ndepth)):
            new_samps.append(cur_depth + low_bound + step * i + eps)

        depth_range_samples = torch.cat(new_samps, 1)
    return depth_range_samples


def samples(cur_depth,exp_var,  ndepth):

        a=torch.Tensor(1,1,1056,1440).uniform_(-1,1).cuda()
        cur_depth=cur_depth+a*0.6
        
        low_bound = -torch.min(cur_depth, exp_var)
        high_bound = exp_var


        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []
        new_samps_2d = []
        x = []

        for i in range(int(ndepth)):
            new_samps.append(cur_depth + low_bound + step * i + eps)


        step_2d = (high_bound - low_bound) / (float(ndepth*2) - 1)
        propotion = ((torch.arange(ndepth*2)).float()/(ndepth*2-1)).view(1,-1,1,1).repeat(1,1,cur_depth.shape[2],cur_depth.shape[3])

        for i in range(int(ndepth*2)):
            new_samps_2d.append(cur_depth + low_bound + step_2d * i + eps)
            x.append(propotion[:,i,:,:].unsqueeze(1))
        depth_range_samples = torch.cat(new_samps, 1)
        depth_range_samples_2d = torch.cat(new_samps_2d, 1)
        x = torch.cat(x, 1)

        return depth_range_samples, x, depth_range_samples_2d


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Conv3dUnit(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv3dUnit(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(Deconv2dBlock, self).__init__()

        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                                   bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2dUnit(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                               bn=bn, relu=relu, bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class FeatExtNet(nn.Module):
    def __init__(self, base_channels):
        super(FeatExtNet, self).__init__()

        self.base_channels = base_channels


        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]


        self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)
        self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3)

        self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
        self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
        self.out_channels.append(2 * base_channels)
        self.out_channels.append(base_channels)



    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)

        outputs["stage1"] = out

        intra_feat = self.deconv1(conv1, intra_feat)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = self.deconv2(conv0, intra_feat)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, base_channels, padding=1)

        self.conv1 = Conv3dUnit(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3dUnit(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3dUnit(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3dUnit(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3dUnit(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3dUnit(base_channels * 8, base_channels * 8, padding=1)

        self.deconv7 = Deconv3dUnit(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.deconv8 = Deconv3dUnit(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.deconv9 = Deconv3dUnit(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x

class RelationEncoder_LSTM(nn.Module):
    # def __init__(self, relation_embedding_dim, rnn_hidden_dim, rel_encoder_method, entityencoder_att_method, use_cuda_wyl):
    def __init__(self, relation_embedding_dim, rnn_hidden_dim, use_cuda_wyl):
        super(RelationEncoder_LSTM, self).__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.use_cuda_wyl = use_cuda_wyl
        # self.rel_encoder_method = rel_encoder_method
        # self.entityencoder_att_method = entityencoder_att_method
        self.rnn = nn.LSTM(relation_embedding_dim, rnn_hidden_dim, batch_first=True)
        self.sdf = nn.Conv1d(50, 2, 1, bias=False)
        self.sdf1 = nn.Conv1d(2, 1, 1, bias=False)


    def init_hidden(self, batch_size):
        # Hidden state axes semantics are (seq_len, batch, rnn_hidden_dim), even when LSTM is set to batch first
        hidden_state = torch.FloatTensor(1, batch_size, self.rnn_hidden_dim)
        hidden_state.copy_(torch.zeros(1, batch_size, self.rnn_hidden_dim))
        cell_state = torch.FloatTensor(1, batch_size, self.rnn_hidden_dim)
        cell_state.copy_(torch.zeros(1, batch_size, self.rnn_hidden_dim))
        if self.use_cuda_wyl == True:
            return (hidden_state.cuda(), cell_state.cuda())
        else:
            return (hidden_state, cell_state)

    def forward(self, relation_embeds):
        # relation_embeds: [num_ent_pairs x num_paths, num_steps, num_feats]
        reshaped_batch_size, num_steps, num_feats = relation_embeds.shape
        h, c = self.init_hidden(reshaped_batch_size)

        self.rnn.flatten_parameters()
        output, (last_hidden, _) = self.rnn(relation_embeds, (h, c))
        # last_hidden: [1, num_ent_pairs x num_paths, rnn_hidden_dim]
        last_hidden = last_hidden.squeeze(dim=0)
        output=output.transpose(1,2)
        sdf_output = self.sdf(output)
        sdf_output = self.sdf1(sdf_output)

        return last_hidden, sdf_output



class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 bn=F,
                 bn_momentum=0.1):
        """Multilayer perceptron shared on resolution (1D or 2D)

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            bn (bool): whether to use batch normalization
        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels

        if ndim == 1:
            mlp_module = Conv1d
        elif ndim == 2:
            mlp_module = Conv1d
        else:
            raise ValueError()

        for ind, out_channels in enumerate(mlp_channels):
            self.append(mlp_module(in_channels, out_channels, 1,
                                   relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for module in self:
            x = module(x)
        return x

class Conv1d(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes.
    optionally followed by batch normalization and relu activation

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=(not bn), **kwargs)
        self.bn = nn.InstanceNorm1d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def set_bn(model, momentum):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def init_uniform(module):
    if module.weight is not None:
        # nn.init.kaiming_uniform_(module.weight)
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


class CoarseMVSNet(nn.Module):
    def __init__(self,ds_ratio,stage_configs,cost_regularization,lamb):
        super(CoarseMVSNet, self).__init__()

        self.stage_configs = stage_configs

        self.lamb = lamb
        self.ds_ratio = ds_ratio

        self.cost_regularization = cost_regularization

    def forward_depth(self,depth, cur_depth, exp_var,proj_matrices, depth_values,features,img,outputs):

        for stage_idx in range(2):
          
            # with torch.no_grad():

                features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
                proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
                stage_scale = self.ds_ratio["stage{}".format(stage_idx + 1)]
                cur_h = img.shape[2] // int(stage_scale)
                cur_w = img.shape[3] // int(stage_scale)

                if depth is not None:

                    cur_depth = depth.detach()
                    exp_var = exp_var.detach()
                    cur_depth = F.interpolate(cur_depth.unsqueeze(1),[cur_h, cur_w], mode='bilinear')
                    exp_var = F.interpolate(exp_var.unsqueeze(1), [cur_h, cur_w], mode='bilinear')

                else:
                    cur_depth = depth_values

                depth_range_samples = uncertainty_aware_samples(cur_depth=cur_depth,
                                                                        exp_var=exp_var,
                                                                        ndepth=self.stage_configs[stage_idx],
                                                                        dtype=img[0].dtype,
                                                                        device=img[0].device,
                                                                        shape=[img.shape[0], cur_h, cur_w])

                outputs_stage = compute_depth(features_stage, proj_matrices_stage,
                                                    depth_samps=depth_range_samples,
                                                    cost_reg=self.cost_regularization[stage_idx],
                                                    lamb=self.lamb,
                                                    is_training=self.training)

                depth = outputs_stage['depth']
                exp_var = outputs_stage['variance']

                outputs["stage{}".format(stage_idx + 1)] = outputs_stage
        return outputs
    def forward_epipolar(self,volume_variance):
        prob_volume_pre=self.cost_regularization[2](volume_variance)
        return prob_volume_pre

def compute_depth(feats, proj_mats, depth_samps, cost_reg, lamb, is_training=False):
 
    with torch.no_grad():

        proj_mats = torch.unbind(proj_mats, 1)
        num_views = len(feats)
        num_depth = depth_samps.shape[1]

        assert len(proj_mats) == num_views, "Different number of images and projection matrices"

        ref_feat, src_feats = feats[0], feats[1:]
        ref_proj, src_projs = proj_mats[0], proj_mats[1:]

        ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        #todo optimize impl
        for src_fea, src_proj in zip(src_feats, src_projs):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])

            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_samps)

            if is_training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2) #in_place method
            del warped_volume
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        prob_volume_pre = cost_reg(volume_variance).squeeze(1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_samps)


        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                            stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                              dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=num_depth - 1)
        prob_conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        samp_variance = (depth_samps - depth.unsqueeze(1)) ** 2
        exp_variance = lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5

    return {"depth": depth, "confidence": prob_conf, 'variance': exp_variance}


