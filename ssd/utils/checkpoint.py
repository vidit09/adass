import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel

from ssd.utils.model_zoo import cache_url


class CheckPointer:
    _last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self,
                 model,
                 optimizer=None,
                 scheduler=None,
                 save_dir="",
                 save_to_disk=None,
                 logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        if isinstance(self.model, DistributedDataParallel):
            data['model'] = self.model.module.state_dict()
        else:
            data['model'] = self.model.state_dict()
#        for p,v in self.model.named_parameters():
#            if id(v) not in list(self.optimizer.state_dict()['state'].keys()):
#                print(p)
#         import pdb;pdb.set_trace()
        data.update(kwargs)
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()


        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = self.model.module

        if 'model_init.pth' in f:
            model.load_state_dict(checkpoint.pop("model"), strict=False)
            # dict = torch.load(
            #     '/cvlabdata2/home/vidit/cross-domain-det/SSD/outputs/vggfpn_ssd512_cityscapescaronly_sagn_multi_attn_50k_attnafterfpn/model_010000.pth')
            # dict = torch.load('/cvlabdata2/home/vidit/cross-domain-det/SSD/outputs/vggfpn_ssd512_sim_cityscapes_sagn_dalmulti_twolayer_noinit_featR_const_005_diff_lr_nodata_augmentation_multi_attn_gndc_featRdet_noPL_moddc_change12k_lrchange10k_flipaug_correcteddecay_dcweightedattn_dck3_diffattn_inittrainedsource/model_init.pth')


            # model.backbone.load_state_dict(dict, strict=False)

            # for k, v in self.model.backbone.state_dict().items():
            #     if '_T' in k:
            #         continue
            #     self.model.backbone.state_dict()[k].copy_(
            #         dict['model']['backbone.' + k])
            #
            # for k, v in self.model.box_head.state_dict().items():
            #     self.model.box_head.state_dict()[k].copy_(
            #         dict['model']['box_head.' + k])

            # model.box_head.load_state_dict(dict, strict=False)
            checkpoint['iteration'] = 0
            return checkpoint
        else:
            model.load_state_dict(checkpoint.pop("model"))

        # import pdb;pdb.set_trace()
        # for p, v in self.model.named_parameters():
        #     if id(v) not in list(checkpoint['optimizer']['state'].keys()):
        #        print(p)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        return torch.load(f, map_location=torch.device("cpu"))
