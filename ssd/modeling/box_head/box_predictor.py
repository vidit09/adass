import torch
from torch import nn

from ssd.layers import SeparableConv2d
from ssd.modeling import registry


class BoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features,return_maps=False):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())



        if return_maps:
            b,hb,wb,a = cls_logits[0].shape
            logs = []
            for clog in cls_logits:
                b,h,w,a = clog.shape
                # import pdb;
                # pdb.set_trace()
                mclog = clog.reshape(b,h,w,-1,self.cfg.MODEL.NUM_CLASSES).softmax(-1)[:,:,:,:,1:]

                mclog = mclog.max(-1)[0].max(-1)[0]
                # mclog = mclog.unsqueeze(1)
                # mclog = torch.nn.functional.interpolate(mclog,size=(hb,wb))
                # logs.append(mclog)
                logs.append(mclog.max(-1)[0].max(-1)[0].unsqueeze(0))

            # import matplotlib.pyplot as plt
            # for ii,l in enumerate(logs):
            #     print(l[0].min(),l[0].max())
            #     plt.imshow(l[0].squeeze(0).detach().cpu().numpy());plt.savefig(f'm0{ii}.png')

            # logs = torch.stack(logs,1).squeeze(2).max(1)[0]
            logs = torch.cat(logs,0)

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        if return_maps:
            return cls_logits, bbox_pred, logs
        else:
            return cls_logits, bbox_pred


@registry.BOX_PREDICTORS.register('SSDBoxPredictor')
class SSDBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


@registry.BOX_PREDICTORS.register('SSDLiteBoxPredictor')
class SSDLiteBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


class BoxPredictorCosine(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        self.cls_weights = nn.ModuleList()
        self.cls_weights_T = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            headers, weights, weights_T = self.cls_block(level, out_channels, boxes_per_location)
            self.cls_headers.append(headers)
            self.cls_weights.append(weights)
            self.cls_weights_T.append(weights_T)
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features,domain='s'):
        cls_logits = []
        bbox_pred = []
        if domain == 's':
            cls_weights = self.cls_weights
        else:
            cls_weights = self.cls_weights_T

        for feature, cls_header, cls_weights, reg_header in zip(features, self.cls_headers, cls_weights, self.reg_headers):
            headers = cls_header(feature).permute(0, 2, 3, 1).contiguous()
            headers = headers / headers.norm(dim=-1,keepdim=True)
            cls_weights = list(cls_weights.parameters())[0]
            cls_weights = cls_weights/ cls_weights.norm(dim=0,keepdim=True)

            cls_logits.append(torch.matmul(headers,cls_weights))
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        if domain == 's':
            return cls_logits, bbox_pred, self.cls_weights
        else :
            return  cls_logits, bbox_pred, self.cls_weights_T


@registry.BOX_PREDICTORS.register('SSDBoxPredictorCosine')
class SSDBoxPredictorCosine(BoxPredictorCosine):
    def cls_block(self, level, out_channels, boxes_per_location):
        cls_weights = nn.ParameterList([nn.Parameter(torch.rand(out_channels//4,boxes_per_location*self.cfg.MODEL.NUM_CLASSES))])
        cls_weights_T = nn.ParameterList([nn.Parameter(torch.rand(out_channels//4,boxes_per_location*self.cfg.MODEL.NUM_CLASSES))])
        return nn.Conv2d(out_channels, out_channels//4, kernel_size=3, stride=1, padding=1), cls_weights, cls_weights_T

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)

def make_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)
