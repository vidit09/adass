import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.layers import L2Norm
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd.modeling import position_encoding

import copy

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}


# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
def add_vgg(cfg, batch_norm=False, group_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif group_norm:
                layers += [conv2d, nn.GroupNorm(32,v//32), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_vgg_bn(cfg, batch_norm=False, group_norm=False ):
    layers = []
    in_channels = 3
    norms = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            if batch_norm:
                norms.append(nn.BatchNorm2d(v))
            elif group_norm:
                norms.append(nn.GroupNorm(v//32, v))

            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers, norms


def add_extras(cfg, i, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.l2_norm = L2Norm(512, scale=20)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []

        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        features.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        return tuple(features)


@registry.BACKBONES.register('vgg')
def vgg(cfg, pretrained=True):
    model = VGG(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.normalize_before = normalize_before

    def _get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):

        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attnw = self.self_attn(q, k, value=src2, attn_mask=None,
                                     key_padding_mask=None)

        # import pdb;pdb.set_trace()
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src, attnw


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode="nearest")

            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7P8P9):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)

class LastLevelP6P7P8P9(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7P8P9, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.p8 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.p9 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7, self.p8, self.p9]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p8 = self.p8(F.relu(p7))
        p9 = self.p9(F.relu(p8))
        return [p6, p7, p8, p9]




class VGGSelfAttnFPNTFP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]
        attn_head = cfg.ATTN_HEAD
        dim_fwd = cfg.ATTN_FWD

        self.unmodulated_feats = cfg.UNMODFEATS

        if not cfg.GN:
            self.vgg, self.norms = add_vgg_bn(vgg_config, batch_norm=True)
        else:
            self.vgg, self.norms = add_vgg_bn(vgg_config, batch_norm=False, group_norm=True)
        self.norms = nn.ModuleList(self.norms)
        self.vgg = nn.ModuleList(self.vgg)

        if cfg.INPUT.IMAGE_SIZE == 300:
            self.pos = position_encoding.build_position_encoding(cfg, indices=50)
        else:
            # self.pos = position_encoding.build_position_encoding(cfg, indices=70)
            self.pos = position_encoding.build_position_encoding(cfg, indices=70,N_steps=128)
            self.pos2 = position_encoding.build_position_encoding(cfg, indices=70, N_steps=512)

        if cfg.MORELAYER:
            self.transformerlayer1 = TransformerEncoderLayer(d_model=512, nhead=8)
            self.transformerlayer2 = TransformerEncoderLayer(d_model=512, nhead=8)
            self.transformerlayer3 = TransformerEncoderLayer(d_model=1024, nhead=8)

            self.transformerlayer1 = TransformerEncoderLayer(d_model=256, nhead=8)
            self.transformerlayer2 = TransformerEncoderLayer(d_model=256, nhead=8)
            self.transformerlayer3 = TransformerEncoderLayer(d_model=256, nhead=8)

            transformerlayer1 = VGGSelfAttnFPNTFP._get_clones(TransformerEncoderLayer(d_model=256, nhead=attn_head, dim_feedforward=dim_fwd),2)
            transformerlayer2= VGGSelfAttnFPNTFP._get_clones(TransformerEncoderLayer(d_model=256, nhead=attn_head, dim_feedforward=dim_fwd),2)
            transformerlayer3= VGGSelfAttnFPNTFP._get_clones(TransformerEncoderLayer(d_model=256, nhead=attn_head, dim_feedforward=dim_fwd),2)
            self.transformerlayers = nn.ModuleList([transformerlayer1,transformerlayer2,transformerlayer3])

        else:
            self.transformerlayers =  nn.ModuleList([TransformerEncoderLayer(d_model=256, nhead=attn_head, dim_feedforward=dim_fwd),TransformerEncoderLayer(d_model=256, nhead=attn_head,dim_feedforward=dim_fwd),
                                                 TransformerEncoderLayer(d_model=256, nhead=attn_head,dim_feedforward=dim_fwd)])
        if cfg.DIFF_ATTN:
            self.diff_attn = True
            self.transformerlayers_T = nn.ModuleList(
                [TransformerEncoderLayer(d_model=256, nhead=8), TransformerEncoderLayer(d_model=256, nhead=8),
                 TransformerEncoderLayer(d_model=256, nhead=8)])
            self.pos_T = position_encoding.build_position_encoding(cfg, indices=70, N_steps=128)
        else:
            self.diff_attn = False

        self.fpn = FPN(in_channels_list=[512, 512, 1024],
                       out_channels=256,
                       conv_block=self.conv_with_kaiming_uniform(
                           False, True  # gn and relu
                       ),
                       top_blocks=LastLevelP6P7P8P9(1024, 256),
                       )

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):
        # self.vgg.load_state_dict(state_dict, strict=False)
        self.vgg.load_state_dict(state_dict)


    def _get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def see_attn(self, ind, weights):
        w = weights[0][ind].reshape(38, 38)
        return w

    def conv_with_kaiming_uniform(self, use_gn=False, use_relu=False):
        def make_conv(
                in_channels, out_channels, kernel_size, stride=1, dilation=1
        ):
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
                bias=False if use_gn else True
            )
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if not use_gn:
                nn.init.constant_(conv.bias, 0)
            module = [conv, ]
            if use_gn:
                module.append(group_norm(out_channels))
            if use_relu:
                module.append(nn.ReLU(inplace=True))
            if len(module) > 1:
                return nn.Sequential(*module)
            return conv

        return make_conv

    def forward(self, x, return_layer=[22],domain='s'):
        features = []

        x = self.vgg[0](x)
        prev = self.vgg[0]
        curr_norm = 0

        returned_layer = []

        for i in range(1, 23):
            if isinstance(prev, nn.Conv2d):
                x = self.norms[curr_norm].to(x.device)(x)
                curr_norm += 1
            x = self.vgg[i](x)

            returned_layer.append(x)
            prev = self.vgg[i]

        returned_layer_ = []
        for ri in return_layer:
            returned_layer_.append(returned_layer[ri - 1])
        returned_layer = returned_layer_[::-1]

        # s = self.l2_norm(x)  # Conv4_3 L2 normalization
        # features.append(s)


        features.append(x)
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
            if i == 29:
                features.append(x)
        x = F.max_pool2d(x, 2)
        features.append(x)
        features = tuple(features)
        features = list(self.fpn(features))
        unmodulated_feats = []
        if self.unmodulated_feats:
            for f in features:
                unmodulated_feats.append(f.clone())
        attns = []
        for ind in range(3):
            src = features[ind]
            ## change the condition below to allow the usage of same pos_embed for 's' and 't' when diff attn is False
            if domain == 's' or not self.diff_attn:
                pos_embed = self.pos(src).flatten(2).permute(2, 0, 1)
            elif self.diff_attn and domain == 't':
                pos_embed = self.pos_T(src).flatten(2).permute(2, 0, 1)
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)

            if self.diff_attn and domain == 't':
                src, attn = self.transformerlayers_T[ind](src, pos_embed)
            else:
                if isinstance(self.transformerlayers[ind],nn.ModuleList):
                    for ll in self.transformerlayers[ind]:
                        src, attn = ll(src, pos_embed)
                else:
                    src, attn = self.transformerlayers[ind](src, pos_embed)
            attns.append(attn)
            features[ind] = src.permute(1, 2, 0).view(bs, c, h, w)

        features = tuple(features)
        return features, tuple([attns]), unmodulated_feats

@registry.BACKBONES.register('vggselfattnfpntfp')
def vgg(cfg, pretrained=True):
    model = VGGSelfAttnFPNTFP(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model


