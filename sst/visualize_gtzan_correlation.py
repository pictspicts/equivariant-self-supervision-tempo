# -*- coding: utf-8 -*-
"""
Pre-trained Log-Tempo Classification Visualization Script for GTZAN
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import tqdm
import sys
import math

# sstモジュールを読み込めるように親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SSTモジュールのインポート
from sst.dataloader_audiofiles import DatasetAudioFiles
from sst.models.tcn import TCN
from sst.models.frontend import FrontEndNoAug
from sst.utils.utils import get_model_filepaths, pack_model_data_paths
from sst.eval import _split_and_batch

TEMPO_RANGE = (0, 300)
TEMPO_MIN = 20
TEMPO_MAX = 300
NUM_CLASSES = TEMPO_RANGE[1] - TEMPO_RANGE[0]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, 'configs', 'eval.yaml')

def bin_to_tempo(bin_idx):
    """ 対数テンポ軸のビン（クラスインデックス）を実際のテンポ(BPM)に変換する """
    log_tempo_min = math.log(TEMPO_MIN)
    log_tempo_max = math.log(TEMPO_MAX)
    log_bin_width = (log_tempo_max - log_tempo_min) / NUM_CLASSES
    return math.exp(log_tempo_min + bin_idx * log_bin_width)

def visualize_gtzan():
    # 1. コンフィグの設定
    config = OmegaConf.load(DEFAULT_CONFIG)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データローダーの準備 (GTZANに切り替え)
    # GTZANデータセットを強制指定
    config.dataset.indexes = ['../datasets_indexes/gtzan.json']
        
    dataset = DatasetAudioFiles(config.dataset)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.dataset.num_workers)

    # 3. 事前学習済みモデルのロード
    print('Loading pre-trained model...')
    
    pretrained_model_dir = config.pretrained_model_dir
        
    pretrained_model_filepath, pretrained_config_filepath = get_model_filepaths(pretrained_model_dir)
    model_data = pack_model_data_paths(pretrained_model_filepath, pretrained_config_filepath)
    config_pretrain = OmegaConf.load(model_data.config_filepath)
    
    # configに事前学習の設定をマージ
    config = OmegaConf.merge(config_pretrain, config)

    # 分類モデルとして初期化
    pre_trained_model = TCN(
        num_filters=config.model.num_filters,
        tempo_range=TEMPO_RANGE, 
        mode='classification', # 強制的に多クラス分類モード
        dropout_rate=config.model.dropout,
        add_proj_head=config.model.add_proj_head,
        proj_head_dim=config.model.proj_head_dim
    )
    
    # 事前学習済み重みのロード
    pre_trained_model.load_state_dict(torch.load(model_data.model_filepath, map_location=device), strict=False)
    pre_trained_model.eval()
    pre_trained_model.to(device)

    # 4. フロントエンド（メルスペクトログラム変換）
    frontend = FrontEndNoAug(config.frontend)
    frontend.to(device)

    # 5. 特徴量抽出ループ
    predicted_tempos = []
    bpms_list = []
    
    max_songs = 1000  # 全楽曲を処理できるように大きく設定
    print(f'Extracting pseudo-tempos for GTZAN dataset...')
    
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if i >= max_songs:
                break
            
            input_audio, tempo_true = data[0].to(device), data[1].to(device)
            inputs = _split_and_batch(input_audio, config.dataset.model_input_num_samples)
            tempi = tempo_true.repeat(inputs.shape[0], tempo_true.shape[1])
            
            mel_inputs, tempi, ts_rate = frontend(inputs, tempi)
            
            # 分類モデル推論 (logits shape: [num_splits, 300])
            z_logits = pre_trained_model(mel_inputs)
            
            # クラスごとにSoftmaxで確率にしてから時間方向に平均をとる
            probs = F.softmax(z_logits, dim=-1) # shape: [num_splits, 300]
            mean_probs = torch.mean(probs, dim=0) # shape: [300]
            
            # 一番確率の高いピーク（ビン）を取得
            argmax_bin = torch.argmax(mean_probs).item()
            
            # 計算したピークを実際のBPMに変換
            pseudo_tempo = bin_to_tempo(argmax_bin)
            
            predicted_tempos.append(pseudo_tempo)
            bpms_list.append(tempo_true.item())

    predicted_np = np.array(predicted_tempos)
    bpms_np = np.array(bpms_list)

    # 統計情報の表示
    print(f"Model outputs stats: min={np.min(predicted_np):.2f}, max={np.max(predicted_np):.2f}, mean={np.mean(predicted_np):.2f}")

    # 7. プロットと保存
    plt.figure(figsize=(10, 8))
    
    # 散布図のプロット
    scatter = plt.scatter(bpms_np, predicted_np, c=bpms_np, cmap='viridis', alpha=0.8, edgecolors='k', s=60)
    plt.colorbar(scatter, label='True BPM')
    
    plt.title('Self-Supervised Classification Model Output vs True BPM (GTZAN)')
    plt.xlabel('True BPM')
    plt.ylabel('Predicted Pseudo-Tempo (BPM)')
    
    # 相関係数の計算
    correlation = np.corrcoef(bpms_np, predicted_np)[0, 1]
    
    # y=x の線（理想的な一致）
    min_bpm = min(np.min(bpms_np), np.min(predicted_np)) - 10
    max_bpm = max(np.max(bpms_np), np.max(predicted_np)) + 10
    plt.plot([min_bpm, max_bpm], [min_bpm, max_bpm], 'k--', alpha=0.5, label='y = x (Perfect Match)')
    
    # オクターブエラー（倍テン・半テン）の線
    plt.plot([min_bpm, max_bpm/2], [min_bpm*2, max_bpm], 'r--', alpha=0.3, label='Double Tempo (x2)')
    plt.plot([min_bpm*2, max_bpm], [min_bpm, max_bpm/2], 'b--', alpha=0.3, label='Half Tempo (x0.5)')
    
    plt.xlim(min(bpms_np)-10, max(bpms_np)+10)
    plt.ylim(min(predicted_np)-10, max(predicted_np)+10)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存
    outdir = config.docker.outdir if "docker" in config and "outdir" in config.docker else SCRIPT_DIR
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'gtzan_correlation.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'Visualization saved to {out_path}')

    print(f"\n======================================")
    print(f"BPMと予測テンポの相関係数 (R): {correlation:.3f}")
    print(f"======================================\n")

if __name__ == "__main__":
    visualize_gtzan()
