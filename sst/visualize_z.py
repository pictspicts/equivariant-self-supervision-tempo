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
import sys

# sstモジュールを読み込めるように親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SSTモジュールのインポート
from sst.dataloader_audiofiles import DatasetAudioFiles
from sst.models.tcn import TCN
from sst.models.frontend import FrontEndNoAug
from sst.utils.utils import get_model_filepaths, pack_model_data_paths
from sst.eval import _split_and_batch

TEMPO_RANGE = (0, 300)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, 'configs', 'eval.yaml')

def visualize():
    # 1. コンフィグの設定
    config = OmegaConf.load(DEFAULT_CONFIG)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データローダーの準備
    # 強制的にローカルのデータセットパスに書き換える（Docker用パスになっている場合への対処）
    if hasattr(config.dataset, 'basedir') and config.dataset.basedir.startswith('/opt/ml/code'):
        config.dataset.basedir = config.dataset.basedir.replace('/opt/ml/code', './')
    
    if hasattr(config.dataset, 'indexes'):
        new_indexes = []
        for idx in config.dataset.indexes:
            if isinstance(idx, str) and idx.startswith('/opt/ml/code'):
                new_indexes.append(idx.replace('/opt/ml/code', '.'))
            else:
                new_indexes.append(idx)
        config.dataset.indexes = new_indexes
        
    dataset = DatasetAudioFiles(config.dataset)
    # ランダムに楽曲を取得するために shuffle=True
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.dataset.num_workers)

    # 3. 事前学習済みモデルのロード
    print('Loading pre-trained model...')
    
    pretrained_model_dir = config.pretrained_model_dir
    if pretrained_model_dir.startswith('/opt/ml/code'):
        pretrained_model_dir = pretrained_model_dir.replace('/opt/ml/code', '.')
        
    pretrained_model_filepath, pretrained_config_filepath = get_model_filepaths(pretrained_model_dir)
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

    # 4. フロントエンド（メルスペクトログラム変換）
    frontend = FrontEndNoAug(config.frontend)
    frontend.to(device)

    # 5. 特徴量抽出ループ
    features_list = []
    bpms_list = []
    
    max_songs = 1000  # 可視化する楽曲数
    print(f'Extracting features for {max_songs} songs...')
    
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if i >= max_songs:
                break
            
            input_audio, tempo_true = data[0].to(device), data[1].to(device)
            inputs = _split_and_batch(input_audio, config.dataset.model_input_num_samples)
            tempi = tempo_true.repeat(inputs.shape[0], tempo_true.shape[1])
            
            mel_inputs, tempi, ts_rate = frontend(inputs, tempi)
            
            # 特徴量抽出 (1D pseudo-tempo)
            z_feats = pre_trained_model(mel_inputs) # shape: (num_splits, 1)
            
            # トラック全体の特徴量として平均をとる
            z_feat_mean = torch.mean(z_feats, dim=0).cpu().numpy()
            
            features_list.append(z_feat_mean)
            bpms_list.append(tempo_true.item())

    features_np = np.array(features_list)
    bpms_np = np.array(bpms_list)

    # 6. 特徴量処理
    print('Processing pseudo-tempo output...')
    z_1d = features_np.flatten()

    # 統計情報の表示
    z_min, z_max = np.min(z_1d), np.max(z_1d)
    z_mean, z_std = np.mean(z_1d), np.std(z_1d)
    print(f"z stats before filtering: min={z_min:.4f}, max={z_max:.4f}, mean={z_mean:.4f}, std={z_std:.4f}")

    # ハズレ値の除去 (1st-99th percentile)
    lower = np.percentile(z_1d, 1)
    upper = np.percentile(z_1d, 99)
    mask = (z_1d >= lower) & (z_1d <= upper)
    
    z_filtered = z_1d[mask]
    bpms_filtered = bpms_np[mask]
    print(f"Filtered {len(z_1d) - len(z_filtered)} outliers (kept {len(z_filtered)} samples)")

    # 7. プロットと保存
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(bpms_filtered, z_filtered, c=bpms_filtered, cmap='coolwarm', alpha=0.8, edgecolors='k', s=80)
    plt.colorbar(scatter, label='True BPM')
    
    plt.title('Pre-trained Pseudo-Tempo $z$ vs True BPM (Filtered Outliers)')
    plt.xlabel('True BPM')
    plt.ylabel('Pseudo-Tempo $z$ (1D Projection)')
    
    # 指定された範囲に設定
    plt.ylim(12, 13)
    
    # 相関係数の再計算（フィルタリング後）
    correlation_filtered = np.corrcoef(bpms_filtered, z_filtered)[0, 1]
    
    # 近似線の追加
    if len(z_filtered) > 1:
        m, b = np.polyfit(bpms_filtered, z_filtered, 1)
        sort_idx = np.argsort(bpms_filtered)
        x_sort = bpms_filtered[sort_idx]
        y_sort = (m*x_sort + b)
        plt.plot(x_sort, y_sort, color='red', linewidth=3, label=f'Trend (R={correlation_filtered:.3f})')
        plt.legend()

    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存パス (Docker所有ディレクトリを避けてスクリプトと同じ場所に保存)
    out_path = os.path.join(SCRIPT_DIR, 'z_visualization.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Visualization saved to {out_path}')

    print(f"\n======================================")
    print(f"BPMと擬似テンポzの相関係数 (R): {correlation_filtered:.3f}")
    print(f"======================================\n")

if __name__ == "__main__":
    visualize()