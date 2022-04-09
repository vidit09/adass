import os
import torch.utils.data
import numpy as np
from PIL import Image

from ssd.structures.container import Container


class CityscapesDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                    'car' )

    def __init__(self, data_dir, ann_file, transform=None, target_transform=None, remove_empty=False):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ann_file = ann_file
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty
        if self.remove_empty:
            # when training, images without annotations are removed.
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}

        self.pseudo_labels = {}
        self.use_pseudo_labels = False
        self.compute_pseudo_labels = False
        self.pl_ids = []

    def __getitem__(self, index):
        if self.use_pseudo_labels:
            image_id = self.pl_ids[index]
            boxes = self.pseudo_labels[image_id]['boxes']/512
            labels = self.pseudo_labels[image_id]['labels']

        else:
            image_id = self.ids[index]
            boxes, labels = self._get_annotation(image_id)

        image = self._read_image(image_id)
        if self.use_pseudo_labels:
            boxes[:,0::2] *= image.shape[1]
            boxes[:,1::2] *= image.shape[0]

        # if self.transform and not self.compute_pseudo_labels:
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def modify_id_list(self,ids):
        old_id = self.ids.copy()
        for id in ids:
            # print(id)
            self.ids.remove(old_id[id])
        self.pl_ids = self.ids
        self.ids = old_id

    def __len__(self):
        if self.use_pseudo_labels:
            return len(self.pl_ids)
        else:
            return len(self.ids)

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        image_id = self.ids[index]
        img_data = self.coco.imgs[image_id]
        return img_data

    def _read_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        datadir = 'leftImg8bit_foggy' if 'foggy' in self.ann_file else 'leftImg8bit'
        if 'train' in self.ann_file :
            image_file = os.path.join(self.data_dir, datadir ,'train', file_name)
        else:
            image_file = os.path.join(self.data_dir, datadir, 'val', file_name)

        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
