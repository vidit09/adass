import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'voc_clipart_train': {
            "data_dir": "clipart",
            "split": "train"
        },
        'voc_clipart_test': {
            "data_dir": "clipart",
            "split": "test"
        },
        'voc_comic_train': {
            "data_dir": "comic",
            "split": "train"
        },
        'voc_comic_test': {
            "data_dir": "comic",
            "split": "test"
        },
        'voc_watercolor_train': {
            "data_dir": "watercolor",
            "split": "train"
        },
        'voc_watercolor_test': {
            "data_dir": "watercolor",
            "split": "test"
        },
        'sbb_train':{
            "data_dir": "sbb",
            "split": "train"
        },
        'sim10k_train':{
            "data_dir": "Sim10k",
            "split": "trainval10k_caronly"
        },
        'cityscapes_train':{
            "data_dir": "Cityscapes",
            "ann_file": "cocoAnnotations/cityscapes_train_caronly_cocostyle.json"
        },
        'cityscapes_val': {
            "data_dir": "Cityscapes",
            "ann_file": "cocoAnnotations/cityscapes_val_caronly_cocostyle.json"
        },
        'kitti_train':{
            "data_dir": "Kitti",
            "split": "train_caronly"
        },
        'cityscapes_train_all': {
            "data_dir": "Cityscapes",
            "ann_file": "cocoAnnotations/cityscapes_train_cocostyle.json"
        },
        'cityscapes_foggy_train': {
            "data_dir": "Cityscapes",
            "ann_file": "cocoAnnotations/cityscapes_foggy_train_cocostyle.json"
        },

        'cityscapes_foggy_val': {
            "data_dir": "Cityscapes",
            "ann_file": "cocoAnnotations/cityscapes_foggy_val_cocostyle.json"
        },
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(    
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)
        elif "sim10k" in name:
            root = DatasetCatalog.DATA_DIR

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="Sim10kDataset", args=args)
        elif "cityscapes" in name:
            root = DatasetCatalog.DATA_DIR

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                ann_file=os.path.join(root, attrs["data_dir"], attrs["ann_file"]),
            )
            return dict(factory="CityscapesDataset", args=args)
        elif "kitti" in name:
            root = DatasetCatalog.DATA_DIR

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="KittiDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
