# -*- coding: utf-8 -*-
"""
Warm-up Script using Margin Ranking Loss on relative tempo labels.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import itertools
import tqdm

from sst.dataloader_audiofiles import DatasetAudioFiles
from sst.models.tcn import TCN
from sst.models.frontend import FrontEndNoAug
from sst.models.finetune import FeatureExtractor
from sst.utils.utils import get_model_filepaths, pack_model_data_paths
from sst.eval import _split_and_batch

TEMPO_RANGE = (0, 300)
DEFAULT_CONFIG = './configs/eval.yaml'

class RankingModel(nn.Module):
    """
    16次元の擬似テンポ z を 1次元のスカラー値に落とし込み、大小を比較できるようにするラッパーモデル
    """
    def __init__(self, base_model):
        super().__init__()
        # 事前学習済みモデルから z を抽出
        self.feat_extractor = FeatureExtractor(base_model, ['tempo_block.dense'])
        # z (16次元) を 相対テンポ (1次元) に変換する線形層
        self.scalar_proj = nn.Linear(16, 1)

    def forward(self, x):
        z_feats = self.feat_extractor(x)['tempo_block.dense']
        # 楽曲分割された入力を平均して、トラック全体の特徴とする
        z_mean = torch.mean(z_feats, dim=0, keepdim=True)
        scalar_tempo = self.scalar_proj(z_mean)
        return scalar_tempo

def warmup():
    config = OmegaConf.load(DEFAULT_CONFIG)
    
    # --- 修正箇所：フロントエンド初期化の「前」に事前学習コンフィグをマージする ---
    pretrained_model_filepath, pretrained_config_filepath = get_model_filepaths(config.pretrained_model_dir)
    model_data = pack_model_data_paths(pretrained_model_filepath, pretrained_config_filepath)
    config_pretrain = OmegaConf.load(model_data.config_filepath)
    config = OmegaConf.merge(config_pretrain, config)
    # -------------------------------------------------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. データセットから10曲だけ取得する
    dataset = DatasetAudioFiles(config.dataset)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.dataset.num_workers)
    
    # ここでは詳細なパラメータ(n_fftなど)がconfigにマージされているのでエラーになりません
    frontend = FrontEndNoAug(config.frontend).to(device)
    
    print("Loading 10 songs for warm-up...")
    songs = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= 10: # 10曲でストップ
                break
            input_audio, tempo_true = data[0].to(device), data[1].to(device)
            inputs = _split_and_batch(input_audio, config.dataset.model_input_num_samples)
            tempi = tempo_true.repeat(inputs.shape[0], tempo_true.shape[1])
            mel_inputs, _, _ = frontend(inputs, tempi)
            songs.append({
                'mel': mel_inputs,
                'bpm': tempo_true.item()
            })

    # 2. 10曲からペアを作成 (10C2 = 45ペア)
    pairs = list(itertools.combinations(songs, 2))
    print(f"Created {len(pairs)} pairs for relative ranking.")

    # 3. 事前学習済みモデルのロード
    base_model = TCN(
        num_filters=config.model.num_filters,
        tempo_range=TEMPO_RANGE, 
        mode=config.model.mode,
        dropout_rate=config.model.dropout,
        add_proj_head=config.model.add_proj_head,
        proj_head_dim=config.model.proj_head_dim
    )
    base_model.load_state_dict(torch.load(model_data.model_filepath, map_location=device), strict=False)
    
    base_model.train()
    base_model.to(device)

    ranking_model = RankingModel(base_model).to(device)
    ranking_model.train()

    # 4. 学習設定
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(ranking_model.parameters(), lr=1e-4)

    # 5. ウォームアップの学習ループ
    epochs = 20
    print("Starting Warm-up Training...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct_order = 0
        
        for song_A, song_B in pairs:
            bpm_A = song_A['bpm']
            bpm_B = song_B['bpm']
            
            if bpm_A == bpm_B:
                continue
            y = torch.tensor([[1.0 if bpm_A > bpm_B else -1.0]]).to(device)
            
            s_A = ranking_model(song_A['mel'])
            s_B = ranking_model(song_B['mel'])
            
            loss = criterion(s_A, s_B, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (s_A.item() > s_B.item() and y.item() == 1.0) or (s_A.item() < s_B.item() and y.item() == -1.0):
                correct_order += 1
                
        acc = correct_order / len(pairs) * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f} - Pairwise Acc: {acc:.1f}%")

    # 6. ウォームアップ済みのモデルを保存
    os.makedirs('./pretrained_model', exist_ok=True)
    save_path = './pretrained_model/TCN_warmed_up.pt'
    torch.save(base_model.state_dict(), save_path)
    print(f"\nWarm-up complete! Model saved to {save_path}")

if __name__ == "__main__":
    warmup()