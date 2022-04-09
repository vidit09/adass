import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    if cfg.SOLVER.TYPE == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.TYPE == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif cfg.SOLVER.TYPE == 'sgd_diff_lr':
        if len(cfg.SOLVER.LR_BLOCK)==1:
            params_list = [{'params': model.backbone.parameters(), 'lr': lr * 0.1},
                           {'params': model.box_head.parameters(), 'lr': lr * 0.1},
                           {'params': model.dclocal.parameters(), 'lr': lr},
                           ]
        elif len(cfg.SOLVER.LR_BLOCK)==3 and not cfg.DIFF_ATTN:
            params_list = [{'params': model.backbone.parameters(), 'lr': float(cfg.SOLVER.LR_BLOCK[0])},
                           {'params': model.box_head.parameters(), 'lr': float(cfg.SOLVER.LR_BLOCK[1])},
                           {'params': model.dclocal.parameters(), 'lr': float(cfg.SOLVER.LR_BLOCK[2])},
                           ]

        elif len(cfg.SOLVER.LR_BLOCK)==3 and  cfg.DIFF_ATTN:
            trainparam = []
            for n,p in model.backbone.named_parameters():
                if 'transformerlayers_T' not in n and 'pos_T' not in n :
                    trainparam.append(p)
            params_list = [{'params': trainparam, 'lr': float(cfg.SOLVER.LR_BLOCK[0])},
                           {'params': model.box_head.parameters(), 'lr': float(cfg.SOLVER.LR_BLOCK[1])},
                           {'params': model.dclocal.parameters(), 'lr': float(cfg.SOLVER.LR_BLOCK[2])},
                           ]
        return torch.optim.SGD(params_list, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


def make_lr_scheduler(cfg, optimizer, milestones=None):
    return WarmupMultiStepLR(optimizer=optimizer,
                             milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
                             gamma=cfg.SOLVER.GAMMA,
                             warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                             warmup_iters=cfg.SOLVER.WARMUP_ITERS)

