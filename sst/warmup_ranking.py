# -*- coding: utf-8 -*-
"""
Warm-up Script using Margin Ranking Loss on relative tempo labels.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
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

class PairwiseFeatureDataset(Dataset):
    """
    事前に抽出した全楽曲の特徴量ペアを返すデータセット
    """
    def __init__(self, z_features, bpms):
        self.z_features = z_features
        self.bpms = bpms
        self.pairs = list(itertools.combinations(range(len(bpms)), 2))
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        z1 = self.z_features[idx1]
        z2 = self.z_features[idx2]
        bpm1 = self.bpms[idx1]
        bpm2 = self.bpms[idx2]
        
        # bpm1 > bpm2 なら 1.0, bpm1 < bpm2 なら -1.0, 同じなら 0.0
        if bpm1 > bpm2:
            y = 1.0
        elif bpm1 < bpm2:
            y = -1.0
        else:
            y = 0.0
            
        return z1, z2, torch.tensor([y], dtype=torch.float32)

def warmup():
    config = OmegaConf.load(DEFAULT_CONFIG)
    
    pretrained_model_filepath, pretrained_config_filepath = get_model_filepaths(config.pretrained_model_dir)
    model_data = pack_model_data_paths(pretrained_model_filepath, pretrained_config_filepath)
    config_pretrain = OmegaConf.load(model_data.config_filepath)
    config = OmegaConf.merge(config_pretrain, config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 事前学習済み models を準備 (重みを固定)
    base_model = TCN(
        num_filters=config.model.num_filters,
        tempo_range=TEMPO_RANGE, 
        mode=config.model.mode,
        dropout_rate=config.model.dropout,
        add_proj_head=config.model.add_proj_head,
        proj_head_dim=config.model.proj_head_dim
    )
    base_model.load_state_dict(torch.load(model_data.model_filepath, map_location=device), strict=False)
    
    # 🚨 base_model の重みを完全に固定 (Freeze)
    for param in base_model.parameters():
        param.requires_grad = False
    
    base_model.eval()
    base_model.to(device)

    ranking_model = RankingModel(base_model).to(device)
    
    frontend = FrontEndNoAug(config.frontend).to(device)
    
    # 2. 全データセットのロードと特徴量の一括抽出
    dataset = DatasetAudioFiles(config.dataset)
    # 特徴量抽出時はバッチサイズ1でループを回す
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.dataset.num_workers)
    
    z_features = []
    bpms = []
    
    print("Extracting features for all songs...")
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            input_audio, tempo_true = data[0].to(device), data[1].to(device)
            inputs = _split_and_batch(input_audio, config.dataset.model_input_num_samples)
            tempi = tempo_true.repeat(inputs.shape[0], tempo_true.shape[1])
            mel_inputs, _, _ = frontend(inputs, tempi)
            
            # 特徴量抽出 (大元のモデルを通すだけ)
            z_feats = ranking_model.feat_extractor(mel_inputs)['tempo_block.dense']
            z_mean = torch.mean(z_feats, dim=0, keepdim=True)
            
            z_features.append(z_mean.cpu())
            bpms.append(tempo_true.item())

    z_features = torch.cat(z_features, dim=0).squeeze(1) # [N, 16] または [N, 1, 16]をsqueeze
    if z_features.dim() == 1:
        z_features = z_features.unsqueeze(0)
    
    # 3. ペアデータセットとデータローダーの作成（バッチ処理）
    pair_dataset = PairwiseFeatureDataset(z_features, bpms)
    pairs_loader = DataLoader(pair_dataset, batch_size=256, shuffle=True)
    print(f"Created {len(pair_dataset)} pairs for relative ranking.")

    # 4. 学習設定
    # 対象を固定されていない parameter に絞る（ここでは scalar_proj のみ）
    ranking_model.scalar_proj.train()
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(ranking_model.scalar_proj.parameters(), lr=1e-4)

    # 5. ウォームアップの学習ループ（バッチ処理）
    epochs = 20
    print("Starting Warm-up Training...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct_order = 0
        valid_pairs = 0
        
        for zA_batch, zB_batch, y_batch in pairs_loader:
            zA_batch = zA_batch.to(device)
            zB_batch = zB_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 同じBPMのペア (y == 0.0) は除外
            mask = (y_batch.squeeze() != 0.0)
            if not mask.any():
                continue
                
            zA_masked = zA_batch[mask]
            zB_masked = zB_batch[mask]
            y_masked = y_batch[mask]
            
            # 線形層を通す
            s_A = ranking_model.scalar_proj(zA_masked)
            s_B = ranking_model.scalar_proj(zB_masked)
            
            loss = criterion(s_A, s_B, y_masked)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_size_valid = zA_masked.size(0)
            total_loss += loss.item() * batch_size_valid
            valid_pairs += batch_size_valid
            
            # ペア関係の正解数を計算
            correct = ((s_A > s_B) == (y_masked == 1.0))
            correct_order += correct.sum().item()
            
        avg_loss = total_loss / valid_pairs if valid_pairs > 0 else 0.0
        acc = (correct_order / valid_pairs) * 100 if valid_pairs > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Pairwise Acc: {acc:.1f}%")

    # 6. ウォームアップ済みのモデルを保存
    os.makedirs('./pretrained_model', exist_ok=True)
    # 線形層の重みを含めるため、今回は ranking_model 自体の state_dict を保存します
    save_path = './pretrained_model/RankingModel_warmed_up.pt'
    torch.save(ranking_model.state_dict(), save_path)
    print(f"\nWarm-up complete! Model saved to {save_path}")

if __name__ == "__main__":
    warmup()