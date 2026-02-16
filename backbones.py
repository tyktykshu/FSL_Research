import torch
import torch.nn as nn
import timm
from timm.models import mobilevit_s, mobilevit_xs, mobilevit_xxs
from args import args
    
class MobileViTWrapper(nn.Module):
    """
    分類器を使用せず特徴ベクトルのみを出力するMobileViTを設する
    """
    def __init__(self, base_model, feature_maps, patch_size=16):
        super(MobileViTWrapper, self).__init__()
        self.base_model = base_model
        self.feature_maps = feature_maps
        self.patch_size = patch_size
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x, mixup=None, lbda=None, perm=None):

        # データ拡張用のMixup処理
        if mixup and lbda is not None and perm is not None:
            x = lbda * x + (1 - lbda) * x[perm]

        # MobileViTの特徴抽出
        x = self.base_model.forward_features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x

def modify_mobilevit(pretrained, feature_maps, variant="xxs"):
    """
    MobileViT のモデルを選択する
    """
    if variant == "s":
        model = mobilevit_s(pretrained=pretrained)
    elif variant == "xs":
        model = mobilevit_xs(pretrained=pretrained)
    elif variant == "xxs":
        model = mobilevit_xxs(pretrained=pretrained)
    else:
        raise ValueError("Invalid MobileViT variant.")
    
    del model.head # 分類ヘッドを削除して特徴抽出部分のみ使用する

    return MobileViTWrapper(model, feature_maps)

def prepareBackbone(feature_maps=32, pretrained=True):
    """
    指定されたバックボーン名に対応するモデルを選択する
    """
    backbone = args.backbone
    return {
        "mobilevit_s": lambda: (modify_mobilevit(pretrained=pretrained, feature_maps=feature_maps, variant="s"), feature_maps),
        "mobilevit_xs": lambda: (modify_mobilevit(pretrained=pretrained, feature_maps=feature_maps, variant="xs"), feature_maps),
        "mobilevit_xxs": lambda: (modify_mobilevit(pretrained=pretrained, feature_maps=feature_maps, variant="xxs"), feature_maps),
        }[backbone.lower()]()

print(" backbones,", end='')