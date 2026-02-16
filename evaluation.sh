#!/bin/bash
# Few-Shot Learning評価用シェルスクリプト
# 指定された保存済みバックボーンモデルを読み込み、Few-Shot評価を実行し、ログを保存します。

# モデル設定
MODEL_NAME="mobilevit_xxs"        # AIモデルの名前
BACKBONE_NAME="${MODEL_NAME}"     # main.pyで使用するバックボーン名

# データセット設定
DATASET="miniimagenet"            # 使用するデータセット名（例: miniimagenet）

# デバイス設定
DEVICE="cuda:0"                   # 使用する計算デバイス（例: cuda:0）

# Few-Shot評価パラメータ
FSW=5          # Few-Shot Ways（1エピソードあたりのクラス数）
FSQ=15         # Few-Shot Queries（各クラスのクエリ数）
FSR=10000      # Few-Shot Runs（実行エピソード数）
FSS=1          # Few-Shot Shots（各クラスのサポートサンプル数）
AS=1           # Augmented Samples (データ拡張数)
FSC="ncm"      # few-shot-classifier (分類器)
FP="ME"        # feature-processing (特徴ベクトル計算用)

# その他パラメータ
RUNS=5
EPOCHS=5
BATCH_SIZE=512
RESOLUTION=224        # 入力画像の解像度（訓練・評価共通）
FEATURE_MAPS=64       # 特徴チャネル数
LR=0.01               # 学習率
GAMMA=0.1             # 学習率減衰率


# 学習済み重みバックボーン（pthファイル）
BACKBONEPATH=""

# データセットパス
DATASETPATH=""

#-------------------- 設定終了 --------------------#


python3 main.py \
    --test-dataset "${DATASET}_test" \
    --backbone "${BACKBONE_NAME}" \
    --runs "${RUNS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --training-dataset "${DATASET}_train" \
    --load-backbone "${BACKBONEPATH}" \
    --test-image-size "${RESOLUTION}" \
    --training-image-size "${RESOLUTION}" \
    --dataset-path "${DATASETPATH}" \
    --few-shot \
    --few-shot-ways "${FSW}" \
    --few-shot-queries "${FSQ}" \
    --few-shot-runs "${FSR}" \
    --few-shot-classifier "${FSC}" \
    --feature-maps "${FEATURE_MAPS}" \
    --few-shot-shots "${FSS}" \
    --device "${DEVICE}" \
    --sample-aug "${AS}"\
    --lr "${LR}" \
    --gamma "${GAMMA}" \
    --use-strides \
    --leaky \
    --feature-processing "${FP}" \
    --freeze-backbone \

