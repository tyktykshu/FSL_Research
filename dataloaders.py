from torchvision import transforms
from args import args
import torch
import os
import json
from PIL import Image
from selfsupervised.selfsupervised import get_ssl_transform
from utils import *
from few_shot_evaluation import EpisodicGenerator
from augmentations import parse_transforms

all_steps = [item for sublist in eval(args.steps) for item in sublist]
supervised = 'lr' in all_steps or 'rotations' in all_steps or 'mixup' in all_steps or 'manifold mixup' in all_steps or (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "" or args.episodic

class DataHolder():
    """
    データとラベルを保持するクラス
    """
    def __init__(self, data, targets, transforms, target_transforms=lambda x:x, opener=lambda x: Image.open(x).convert('RGB')):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert(self.length == len(targets))
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.opener = opener

    def __getitem__(self, idx):
        """
        指定インデックスのデータを取得する関数
        DataLoaderから自動的に呼ばれる
        """
        if isinstance(self.data[idx], str):
            elt = self.opener(args.dataset_path + self.data[idx])
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.target_transforms(self.targets[idx])

    def __len__(self):
        return self.length

class CategoriesSampler():
    """
    Few-Shot Learning用のエピソード制作用クラス

    1エピソード = n_waysクラス x (n_shots + n_queries)枚のインデックスを生成する
    """    
    def __init__(self, datasetName):
        self.batch_size = args.batch_size
        self.generator = EpisodicGenerator(datasetName=datasetName, dataset_path=args.dataset_path)
        self.n_ways = args.few_shot_ways
        self.n_shots = args.few_shot_shots
        self.n_queries = args.few_shot_queries
        self.episodic_iterations_per_epoch = args.episodic_iterations_per_epoch

    def __len__(self):
        return self.episodic_iterations_per_epoch
    
    def __iter__(self):
        """
        1エポック分のエピソードを順番に生成する
        各エピソードではshots → queries の順番でインデックスを並べる
        """
        for _ in range(self.episodic_iterations_per_epoch):
            episode = self.generator.sample_episode(ways=self.n_ways, n_shots=self.n_shots, n_queries=self.n_queries)
            batch = []
            for c, class_idx in enumerate(episode['choice_classes']):
                offset = sum(self.generator.num_elements_per_class[:class_idx])
                batch = batch + [offset+s for s in episode['shots_idx'][c]+episode['queries_idx'][c]]
            batch = torch.tensor(batch)
            yield batch

def dataLoader(dataholder, shuffle, datasetName, episodic):
    """
    DataHolderをPyTorchのDataLoaderに変換する

    episodic=True の場合はFew-Shot用のCategoriesSamplerを使用する
    """
    if episodic : 
        sampler = CategoriesSampler(datasetName=datasetName)
        return torch.utils.data.DataLoader(dataholder, num_workers = min(os.cpu_count(), 8), batch_sampler=sampler)
    return torch.utils.data.DataLoader(dataholder, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(os.cpu_count(), 8))

class TransformWrapper(object):
    """
    複数の変換（supervisedやSSLなど）を辞書形式でまとめて返すクラス
    """
    def __init__(self, all_transforms):
        self.all_transforms = all_transforms

    def __call__(self, image):
        """
        1枚の画像に対して、登録された全ての変換を適用する
        戻り値は辞書型
        """
        out = {}
        for name, T in self.all_transforms.items():
            out[name] = T(image)
        return out

def get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms):
    """
    学習用・テスト用の画像変換を生成する関数。
    
    trainの場合：supervised変換 + self-supervised変換をまとめる
    testの場合：通常のComposeのみ返す
    """
    if datasetName == 'train':
        supervised_transform_str = args.training_transforms if len(args.training_transforms) > 0 else default_train_transforms
        supervised_transform = parse_transforms(supervised_transform_str, image_size) 
        all_transforms = {}
        if supervised:
            all_transforms['supervised'] = transforms.Compose(supervised_transform)
        all_transforms.update(get_ssl_transform(image_size, normalization=supervised_transform[-1]))
        trans = TransformWrapper(all_transforms)
    else:
        trans = transforms.Compose(parse_transforms(args.test_transforms if len(args.test_transforms) > 0 else default_test_transforms, image_size))

    return trans

def miniimagenet(datasetName):
    """
    miniImageNetデータセットを読み込み、
    DataLoaderを作成して辞書形式で返す関数。
    """
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["miniimagenet_" + datasetName]
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 84
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 84
    default_train_transforms = ['randomresizedcrop','colorjitter', 'randomhorizontalflip', 'totensor', 'miniimagenetnorm']
    if args.sample_aug == 1:
        default_test_transforms = ['resize_92/84', 'centercrop', 'totensor', 'miniimagenetnorm']
    else:
        default_test_transforms = ['randomresizedcrop', 'totensor', 'miniimagenetnorm']
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)

    return {"dataloader": dataLoader(DataHolder(dataset["data"], dataset["targets"], trans), shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="miniimagenet_"+datasetName), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def prepareDataLoader(name):
    """
    データセット名から対応するデータローダーを作成する関数。
    複数データセット指定にも対応。
    """
    if isinstance(name, str):
        name = [name]
    result = []
    dataset_options = {
            "miniimagenet_train": lambda: miniimagenet("train"),
            "miniimagenet_validation": lambda: miniimagenet("validation"),
            "miniimagenet_test": lambda: miniimagenet("test"),
        }
    
    for elt in name:
        assert elt.lower() in dataset_options.keys(), f'The chosen dataset "{elt}" is not existing, please provide a valid option: \n {list(dataset_options.keys())}'
        result.append(dataset_options[elt.lower()]())

    return result

# 各データセットを生成
trainSet = prepareDataLoader(args.training_dataset)  if args.training_dataset   != "" else []
validationSet = prepareDataLoader(args.validation_dataset) if args.validation_dataset != "" else []
testSet = prepareDataLoader(args.test_dataset) if args.test_dataset != "" else []

print(" dataloaders")
