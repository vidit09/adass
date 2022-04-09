from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .coco import COCODataset
from .sim10k import Sim10kDataset
from .cityscapes import CityscapesDataset
from .kitti import KittiDataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
    'Sim10kDataset': Sim10kDataset,
    'CityscapesDataset': CityscapesDataset,
    'KittiDataset': KittiDataset
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True, is_da=False):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset  or factory == Sim10kDataset or factory == KittiDataset:
            args['keep_difficult'] = not is_train
        elif factory == COCODataset or factory == CityscapesDataset:
            args['remove_empty'] = is_train
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets or if only one training dataset
    if not is_train or len(datasets) == 1:
        return datasets

    if is_da:
        if factory == VOCDataset:
            dataset1 = ConcatDataset(datasets[:2])
            return [dataset1, datasets[-1]]
        elif factory == Sim10kDataset or factory == CityscapesDataset or factory == KittiDataset:
            return [datasets[0], datasets[1]]

    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
