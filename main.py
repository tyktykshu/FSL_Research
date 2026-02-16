import torch
import random
import math
import time
import torch.nn as nn
import numpy as np
import colorama
from colorama import Style, Fore, Back

print("Loadning...", end="")

import classifiers
import backbones
from utils import *
from args import args
from dataloaders import trainSet, testSet
from few_shot_evaluation import EpisodicGenerator

colorama.init(autoreset = True)

# ランダムシードを生成
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def to(obj, device):
    """
    テンソル（またはその入れ子構造）を指定 device に移す関数

    list / dict の場合は再帰的に中身を .to(device) する
    それ以外は「テンソルである」前提で .to(device) を呼ぶ
    """
    if isinstance(obj, list):
        return [to(o, device) for o in obj]
    elif isinstance(obj, dict):
        return {k:to(v, device) for k,v in obj.items()}
    else:
        return obj.to(device)

def testFewShot(features, datasets = None):
    """
    抽出済み特徴量（features）に対して Few-Shot 評価を実行する

    ・Parameters
    features : list
        generateFeatures() の出力（データセットごとのクラス別特徴量リスト）
        形式は features[dataset_idx][class_idx] = {"name_class": str, "features": Tensor[N, D]}
    datasets : list | None
        dataloaders.py の dataset dict のリスト（表示用に name を使う）
        None の場合は表示しない

    ・Returns
    torch.Tensor
        shape=(num_datasets, 2)
        [:,0] = 平均精度（%）
        [:,1] = 95%信頼区間の半幅（±表記用）
    """
    results = torch.zeros(len(features), 2) 
    for i in range(len(features)):
        accs = []
        feature = features[i]
        Generator = EpisodicGenerator
        generator = Generator(datasetName=None if datasets is None else datasets[i]["name"], num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
        for _ in range(args.few_shot_runs):
            shots = []
            queries = []
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries)
            shots, queries = generator.get_features_from_indices(feature, episode)
            accs.append(classifiers.evalFewShotRun(shots, queries))
        accs = 100 * torch.tensor(accs)
        low, up = confInterval(accs)
        results[i, 0] = torch.mean(accs).item()
        results[i, 1] = (up - low) / 2
        if datasets is not None:
            display(" " * (1 + max(0, len(datasets[i]["name"]) - 16)) + opener + "{:6.2f}% (±{:6.2f})".format(results[i, 0], results[i, 1]) + ender, force = True)
    return results

def process(featuresSet, mean):
    """
    Few-Shot 前に特徴量を前処理する

    args.feature_processing の指定に応じて実行:
    - 'M' : 全特徴から平均ベクトル mean を引く（中心化）
    - 'E' : L2 正規化（単位球面への射影）
    """
    for features in featuresSet:
        if "M" in args.feature_processing:
            for feat in features:
                feat["features"] = feat["features"] - mean.unsqueeze(0)
        if "E" in args.feature_processing:
            for feat in features:
                feat["features"] = feat["features"] / torch.norm(feat["features"], dim = 1, keepdim = True)
    return featuresSet

def computeMean(featuresSet):
    """
    複数データセットの特徴量から、平均ベクトル（D 次元）を計算する

    featuresSet は generateFeatures() の出力（データセットごとのクラス別特徴）
    データセットごとに「全クラスの特徴を結合して平均」を取り、それをさらにデータセット間で平均する
    """
    avg = None
    for features in featuresSet:
        if avg == None:
            avg = torch.cat([features[i]["features"] for i in range(len(features))]).mean(dim = 0)
        else:
            avg += torch.cat([features[i]["features"] for i in range(len(features))]).mean(dim = 0)
    return avg / len(featuresSet)

def generateFeatures(backbone, datasets, sample_aug=args.sample_aug):
    """
    データローダから画像を流して、バックボーン特徴量を抽出する

    ・Parameters
    backbone : nn.Module
        入力画像 -> 特徴ベクトル（B,D）を返すモデル
    datasets : list[dict]
        dataloaders.py で作る dataset dict のリスト
    sample_aug : int
        test 時にランダムクロップ等でデータ拡張を行う場合の拡張数

    ・Returns
    list
        データセットごとのクラス別特徴:
        results[dataset_idx][class_idx] = {"name_class": str, "features": Tensor[N, D]}
    """
    backbone.eval()
    results = []
    for testSetIdx, dataset in enumerate(datasets):
        n_aug = 1 if 'train' in dataset['name'] else sample_aug
        allFeatures = [{"name_class": name_class, "features": []} for name_class in dataset["name_classes"]]
        with torch.no_grad():
            for augs in range(n_aug):
                features = [{"name_class": name_class, "features": []} for name_class in dataset["name_classes"]]
                for batchIdx, (data, target) in enumerate(dataset["dataloader"]):
                    if isinstance(data, dict):
                        data = data["supervised"]
                    data, target = to(data, args.device), target.to(args.device)
                    feats = backbone(data).to("cpu")
                    for i in range(feats.shape[0]):
                        features[target[i]]["features"].append(feats[i])
                for c in range(len(allFeatures)):
                    if augs == 0:
                        allFeatures[c]["features"] = torch.stack(features[c]["features"])/n_aug
                    else:
                        allFeatures[c]["features"] += torch.stack(features[c]["features"])/n_aug

        results.append([{"name_class": allFeatures[i]["name_class"], "features": allFeatures[i]["features"]} for i in range(len(allFeatures))])
    return results

def get_optimizer(parameters, name, lr, weight_decay):
    """
    optimizer を文字列指定で切り替えるための関数
    """
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {name} not supported')

allRunTrainStats = None
allRunValidationStats = None
allRunTestStats = None

# Few-Shot 評価フロー
for nRun in range(args.runs):

    # バックボーンを取得
    import backbones
    backbone, outputDim = backbones.prepareBackbone()
    backbone.load_state_dict(torch.load(args.load_backbone, weights_only=True))
    backbone = backbone.to(args.device)

    # AIモデル（分類器を除く）の学習可能パラメータ数を算出
    if not args.silent:
        numParamsBackbone = torch.tensor([m.numel() for m in backbone.parameters()]).sum().item()
        print("containing {:,} parameters and feature space of dim {:d}.".format(numParamsBackbone, outputDim))
        print("Preparing criterion(s) and classifier(s)... ", end='')

    # 1エポックあたりの学習ステップ数（DataLoader の反復回数）を決める
    try:
        nSteps = torch.min(torch.tensor([len(dataset["dataloader"]) for dataset in trainSet])).item()
        if args.dataset_size > 0 and math.ceil(args.dataset_size / args.batch_size) < nSteps:
            nSteps = math.ceil(args.dataset_size / args.batch_size)
    except:
        nSteps = 0

    # 学習で使う criterion（=広義の損失関数/分類ヘッド）を用意する
    criterion = {}
    all_steps = [item for sublist in eval(args.steps) for item in sublist]
    criterion['supervised'] = [classifiers.prepareCriterion(outputDim, dataset["num_classes"]) for dataset in trainSet]

    # 分類器の学習可能パラメータ数を算出
    numParamsCriterions = 0
    for c in [item for sublist in criterion.values() for item in sublist]:
        c.to(args.device)
        numParamsCriterions += torch.tensor([m.numel() for m in c.parameters()]).sum().item()
    if not args.silent:
        print(" total is {:,} parameters.".format(numParamsBackbone + numParamsCriterions))
        print("Preparing optimizer... ", end='')

    # optimizerに渡す学習可能パラメータのリストを作成
    if not args.freeze_backbone:
        parameters = list(backbone.parameters())
    else:
        parameters = []
    for c in [item for sublist in criterion.values() for item in sublist] :
        parameters += list(c.parameters())

    # 完了ログ
    if not args.silent:
        print(" done.")
        print()

    print(Back.BLUE + "RUNS : " + str(nRun + 1))
    print()

    tick = time.time()
    best_val = 1e10 if not args.few_shot else 0
    lr = args.lr

    # 評価ループ
    for epoch in range(args.epochs):
        print(Back.RED + "EPOCH: " + str(epoch + 1), end="")

        if (epoch == args.warmup_epochs or (epoch in args.milestones)) and len(parameters)>0:
            if args.scheduler == "multistep" and epoch == args.warmup_epochs:
                optimizer = get_optimizer(parameters, args.optimizer.lower(), lr=lr, weight_decay=args.wd)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = [(n-args.warmup_epochs) * nSteps for n in args.milestones], gamma = args.gamma)
            
            if args.scheduler != "multistep":
                optimizer = get_optimizer(parameters, args.optimizer.lower(), lr=lr, weight_decay=args.wd)
                if epoch == args.warmup_epochs:
                    interval = nSteps * (args.milestones[0]-args.warmup_epochs-1)
                else:
                    index = args.milestones.index(epoch)
                    interval = nSteps * (args.milestones[index + 1] - args.milestones[index]-1)
                if args.scheduler == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = interval, eta_min = lr * args.end_lr_factor)
                elif args.scheduler == "linear":
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = optimizer, start_factor = 1, end_factor = args.end_lr_factor, total_iters = interval, last_epoch=-1)
                else:
                    raise ValueError(f"Unknown scheduler {args.scheduler}")
                lr = lr * args.gamma

        continueTest = False
        meanVector = None
        trainStats = None

        if trainSet != []:
            opener = Fore.CYAN
            if (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "":
                if epoch >= args.skip_epochs:
                    featuresTrain = generateFeatures(backbone, trainSet)
                    meanVector = computeMean(featuresTrain) 
                    featuresTrain = process(featuresTrain, meanVector) 
            ender = Style.RESET_ALL
        
        if testSet != [] and epoch >= args.skip_epochs: 
            opener = Fore.RED
            infer_start = time.time()

            if args.few_shot or args.save_features_prefix != "":
                featuresTest = generateFeatures(backbone, testSet)
                featuresTest = process(featuresTest, meanVector)
                tempTestStats = testFewShot(featuresTest, testSet)

            ender = Style.RESET_ALL
        print()