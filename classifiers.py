import torch
import torch.nn as nn
import random
import numpy as np
from args import args

class LR(nn.Module):
    """
    バックボーン（特徴抽出器）から得られた特徴ベクトルを全結合層（Linear層）で分類する
    """
    def __init__(self, inputDim, numClasses, backbone=None):
        super(LR, self).__init__()
        self.fc = nn.Linear(inputDim, numClasses)
        self.fcRotations = nn.Linear(inputDim, 4)
        self.criterion = nn.CrossEntropyLoss() if args.label_smoothing == 0 else LabelSmoothingLoss(numClasses, args.label_smoothing) 
        self.backbone = backbone

    def forward(self, backbone, dataStep, y, lr=False, rotation=False, mixup=False, manifold_mixup=False):
        """
        順伝播（forward計算）を行う関数

        :param y: 正解ラベル
        """
        lbda, perm, mixupType = None, None, None
        loss, score, multiplier = 0., torch.zeros(1), 1

        # mixup処理
        if mixup or manifold_mixup:
            multiplier = 0.5
            perm = torch.randperm(dataStep.shape[0])
            if mixup:
                lbda = random.random()
                mixupType = "mixup"
            else:
                lbda = np.random.beta(2,2)
                mixupType = "manifold mixup"
        yRotations = None

        # 回転処理
        if rotation:
            bs = dataStep.shape[0] // 4
            targetRot = torch.LongTensor(dataStep.shape[0]).to(args.device)
            targetRot[:bs] = 0
            dataStep[bs:] = dataStep[bs:].transpose(3,2).flip(2)
            targetRot[bs:2*bs] = 1
            dataStep[2*bs:] = dataStep[2*bs:].transpose(3,2).flip(2)
            targetRot[2*bs:3*bs] = 2
            dataStep[3*bs:] = dataStep[3*bs:].transpose(3,2).flip(2)
            targetRot[3*bs:] = 3
            yRotations = targetRot
        
        # backboneの呼び出し
        x = backbone(dataStep, mixup = mixupType, lbda = lbda, perm = perm)

        # 予測と正解率の計算
        if lr or mixup or manifold_mixup:
            output = self.fc(x)
            decision = output.argmax(dim = 1)
            score = (decision - y == 0).float().mean()
            loss = self.criterion(output, y)
            multiplier = 0.5
        if lbda is not None:
            loss = lbda * loss + (1 - lbda) * self.criterion(output, y[perm])
            score = lbda * score + (1 - lbda) * (decision - y[perm] == 0).float().mean()
        if yRotations is not None:
            outputRotations = self.fcRotations(x)
            loss = multiplier * (loss + (self.criterion(outputRotations, yRotations) if lbda == None else (lbda * self.criterion(outputRotations, yRotations) + (1 - lbda) * self.criterion(outputRotations, yRotations[perm]))))
        return loss, score

def ncm(shots, queries):
    """
    NCM (Nearest Class Mean) 分類を行う関数
    各クラスのサポートデータ（shots）の平均ベクトル（centroid）を計算し、クエリデータ（queries）とのユークリッド距離を求め、最も近いクラスに分類
    """
    centroids = torch.stack([shotClass.mean(dim = 0) for shotClass in shots])
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        distances = torch.norm(queriesClass.unsqueeze(1) - centroids.unsqueeze(0), dim = 2)
        score += (distances.argmin(dim = 1) - i == 0).float().sum()
        total += queriesClass.shape[0]
    return score / total

def evalFewShotRun(shots, queries):
    """
    Few-Shot評価を実行する関数
    args.few_shot_classifier の設定に応じて対応するFew-Shot分類器を呼び出す（本研究では NCM のみ使用する）
    """
    if args.few_shot_classifier.lower() != "ncm":
        raise ValueError("few_shot_classifier must be 'ncm'")
    with torch.no_grad():
        return ncm(shots, queries)

def prepareCriterion(outputDim, numClasses):
    """
    学習時に使用する分類器（criterion）を準備する関数
    """
    return {
        "lr": lambda: LR(outputDim, numClasses), 
        }[args.classifier.lower()]()

print(" classifiers,", end="")
