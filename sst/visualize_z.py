# -*- coding: utf-8 -*-
"""
Pre-trained Pseudo-Tempo (z) Visualization Script
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import tqdm

# umapを利用する場合は以下をコメントアウト解除してください
# import umap

# SSTモジュールのインポート
from sst.dataloader_audiofiles import DatasetAudioFiles
from sst.models.tcn import TCN
from sst.models.frontend import FrontEndNoAug
from sst.models.finetune import FeatureExtractor
from sst.utils.utils import get_model_filepaths, pack_model_data_paths
from sst.eval import _split_and_batch

TEMPO_RANGE = (0, 300)
DEFAULT_CONFIG = './configs/eval.yaml'

def visualize():
    # 1. コンフィグの設定
    config = OmegaConf.load(DEFAULT_CONFIG)
    
    # # バッチサイズ等を可視化用に調整
    # config.training.batch_size = 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データローダーの準備
    # ACM_mirum_tempo_tiny などの評価用（ラベル付き）データを読み込みます
    dataset = DatasetAudioFiles(config.dataset)
    # ランダムに楽曲を取得するために shuffle=True
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.dataset.num_workers)

    # 3. 事前学習済みモデルのロード
    print('Loading pre-trained model...')
    pretrained_model_filepath, pretrained_config_filepath = get_model_filepaths(config.pretrained_model_dir)
    model_data = pack_model_data_paths(pretrained_model_filepath, pretrained_config_filepath)
    config_pretrain = OmegaConf.load(model_data.config_filepath)
    config = OmegaConf.merge(config_pretrain, config)

    pre_trained_model = TCN(
        num_filters=config.model.num_filters,
        tempo_range=TEMPO_RANGE, 
        mode=config.model.mode,
        dropout_rate=config.model.dropout,
        add_proj_head=config.model.add_proj_head,
        proj_head_dim=config.model.proj_head_dim
    )
    
    # 事前学習済み重みのロード (Projection Headの差異等を吸収するため strict=False)
    pre_trained_model.load_state_dict(torch.load(model_data.model_filepath, map_location=device), strict=False)
    pre_trained_model.eval()
    pre_trained_model.to(device)

    # 潜在表現 (z) を抽出する FeatureExtractor の設定
    # 16次元の出力を持つ 'tempo_block.dense' をターゲットにします
    feat_getter = FeatureExtractor(pre_trained_model, layers=['tempo_block.dense'])
    feat_getter.eval()

    # 4. フロントエンド（メルスペクトログラム変換）
    frontend = FrontEndNoAug(config.frontend)
    frontend.to(device)

    # 5. 特徴量抽出ループ
    features_list = []
    bpms_list = []
    
    max_songs = 50  # 可視化する楽曲数（適宜変更してください）
    print(f'Extracting features for {max_songs} songs...')
    
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if i >= max_songs:
                break
            
            input_audio, tempo_true = data[0].to(device), data[1].to(device)
            inputs = _split_and_batch(input_audio, config.dataset.model_input_num_samples)
            tempi = tempo_true.repeat(inputs.shape[0], tempo_true.shape[1])
            
            mel_inputs, tempi, ts_rate = frontend(inputs, tempi)
            
            # 特徴量抽出
            extracted_feats = feat_getter(mel_inputs)
            z_feats = extracted_feats['tempo_block.dense'] # shape: (num_splits, 16)
            
            # トラック全体の特徴量として平均をとる
            z_feat_mean = torch.mean(z_feats, dim=0).cpu().numpy()
            
            features_list.append(z_feat_mean)
            bpms_list.append(tempo_true.item())

    features_np = np.array(features_list) # (max_songs, 16)
    bpms_np = np.array(bpms_list)

    # 6. 次元削減 (16次元 -> 1次元)
    print('Applying dimensionality reduction...')
    
    # --- PCAを使用する場合 ---
    reducer = PCA(n_components=1)
    z_1d = reducer.fit_transform(features_np).flatten()
    
    # --- UMAPを使用する場合 (コメントアウト解除) ---
    # reducer = umap.UMAP(n_components=1, random_state=42)
    # z_1d = reducer.fit_transform(features_np).flatten()

    # 7. プロットと保存
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(bpms_np, z_1d, c=bpms_np, cmap='coolwarm', alpha=0.8, edgecolors='k', s=80)
    plt.colorbar(scatter, label='True BPM')
    
    plt.title('Pre-trained Pseudo-Tempo $z$ vs True BPM')
    plt.xlabel('True BPM')
    plt.ylabel('Pseudo-Tempo $z$ (1D Projection)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    outdir = config.docker.outdir if hasattr(config, 'docker') else './'
    if getattr(config, 'docker', None) and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
        
    out_path = os.path.join(outdir, 'z_visualization.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Visualization saved to {out_path}')

    # --- visualize_z.py の一番最後に追加 ---
    correlation = np.corrcoef(bpms_np, z_1d)[0, 1]
    print(f"\n======================================")
    print(f"BPMと擬似テンポzの相関係数 (R): {correlation:.3f}")
    print(f"======================================\n")

if __name__ == "__main__":
    visualize()