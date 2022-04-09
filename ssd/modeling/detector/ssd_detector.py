import torch
from torch import nn


from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.modeling.domain_classifier import  DomainClassifierMultiLocalFEReweighted


class SSDDetectorWithDALMultiFER(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)
        self.ins = list(cfg.MODEL.BACKBONE.OUT_CHANNELS)
        self.returnlayer = [22,15]
        self.dclocal = DomainClassifierMultiLocalFEReweighted(self.ins,cfg.FEReWEIGHTING).cuda()

    def forward(self, images, targets=None, constant=0.,domain='s'):
        if self.cfg.MODEL.BACKBONE.NAME in ['vggselfattnfpntfp']:

            features, attnw, unmodulated_feats = self.backbone(images,self.returnlayer,domain)

        else:
            features = self.backbone(images)
            attnw = [None]*len(features)

        source_feat = features

        return_predmaps = self.cfg.MODEL.PREDMAPSLOSS
        if self.training:

            if targets is not None:

                detections, detector_losses, predmaps = self.box_head(source_feat, targets, return_predmaps, domain)
            else:

                if return_predmaps:
                    _, _, predmaps = self.box_head(source_feat, targets, return_predmaps)
                else:
                    predmaps = None

        else:
            detections, detector_losses = self.box_head(source_feat, targets)

        returned_layer = features

        # domain classifier loss
        if self.training:
            targetsd = []
            weights = attnw[0]
            for ii in range(len(returned_layer)):
                feat = returned_layer[ii]
                b,c,h,w = feat.shape
                if targets and domain=='s':
                    targetsd.append(torch.zeros(features[0].size(0),h,w).to(images.device).long())
                elif targets or domain=='t':
                    targetsd.append(torch.ones(features[0].size(0),h,w).to(images.device).long())
            if self.cfg.UNMODFEATS:
                local_domain_loss = self.dclocal(returned_layer,constant,targetsd,weights,predmaps,unmodulated_feats=unmodulated_feats)
            else:
                local_domain_loss = self.dclocal(returned_layer, constant, targetsd, weights, predmaps)
            try:
                detector_losses.update({'domain_loss':local_domain_loss})
            except:
                detector_losses = {}
                detector_losses.update({'domain_loss':local_domain_loss})

        if self.training:
            return detector_losses
        return detections

    def get_attnw(self,images):
        features, attnw, returned_layer = self.backbone(images, self.returnlayer)
        _, _ , predmaps = self.box_head(features, None, True)
        val, ind = torch.max(attnw[0], 1)
        val = (val - val.min(1)[0].unsqueeze(1)) / (val.max(1)[0].unsqueeze(1) - val.min(1)[0].unsqueeze(1))
        val = val.reshape(predmaps.shape)
        return val, predmaps

    def get_cos(self, images):
        features, attnw, returned_layer = self.backbone(images, self.returnlayer)
        weights = attnw[0]
        val, ind = torch.max(weights, 1)
        print(val.shape)
        val = (val - val.min(1)[0].unsqueeze(1)) / (val.max(1)[0].unsqueeze(1) - val.min(1)[0].unsqueeze(1))
        oval = val.reshape(features[0].shape[0], features[0].shape[2], features[0].shape[3])
        val = oval + 1
        val = val.unsqueeze(1)

        fg = features[0]*val
        bg = features[0]*(2-val)

        fg = fg.mean((0,2,3))
        bg = bg.mean((0,2,3))
        return fg,bg,oval

    def get_results(self,images):
        features, attnw, returned_layer = self.backbone(images, self.returnlayer)
        ff = features[0].max(1)[0].detach().cpu().numpy()
        ff1 = features[1].max(1)[0].detach().cpu().numpy()
        ff2 = features[2].max(1)[0].detach().cpu().numpy()

        att = attnw[0][0].max(1)[0].reshape(-1, 64, 64).detach().cpu().numpy()
        att1 = attnw[0][1].max(1)[0].reshape(-1, 32, 32).detach().cpu().numpy()
        att2 = attnw[0][2].max(1)[0].reshape(-1, 16, 16).detach().cpu().numpy()

        return [ff,ff1,ff2,att,att1,att2]



