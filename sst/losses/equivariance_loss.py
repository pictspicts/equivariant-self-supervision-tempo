import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EquivarianceLoss(nn.Module):
    def __init__(self, num_classes=300, tempo_min=20, tempo_max=300, alpha=0.985, device=None):
        """
        等変性損失 (Equivariance Loss)
        確率分布をスカラー値（Z変換 / $\phi(y)$）に特殊圧縮し、物理的なシフト量を直接ペナルティ化する。
        """
        super(EquivarianceLoss, self).__init__()
        self.num_classes = num_classes
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cpu")
        
        # Log bin widthの計算
        self.log_tempo_min = math.log(self.tempo_min)
        self.log_tempo_max = math.log(self.tempo_max)
        self.log_bin_width = (self.log_tempo_max - self.log_tempo_min) / self.num_classes
        
        # アルファの累乗ベクトルを事前計算: [alpha^0, alpha^1, ..., alpha^(N-1)]
        powers = torch.arange(self.num_classes, dtype=torch.float32, device=self.device)
        # bufferとして登録（デバイス間での移動を自動化）
        self.register_buffer('alpha_powers', torch.pow(self.alpha, powers))

    def _projection_phi(self, P):
        """
        確率分布 P の各要素に alpha^i を掛けて合計し、1つの数値に射影する
        P: (batch_size, num_classes) の確率分布
        戻り値: (batch_size,) のスカラーテンソル
        """
        # (batch_size, num_classes) * (1, num_classes) -> dim=-1で合計
        return torch.sum(P * self.alpha_powers.unsqueeze(0), dim=-1)

    def forward(self, z_i, z_j, ts_rate_i, ts_rate_j):
        """
        ロス関数の計算
        z_i, z_j: TCNが出力するロジット (batch_size, num_classes)
        ts_rate_i, ts_rate_j: オリジナルと水増しのタイムストレッチ比率 (batch_size, 1)
        """
        # 分布を確率値（Softmax）に変換
        prob_i = F.softmax(z_i, dim=-1)
        prob_j = F.softmax(z_j, dim=-1)
        
        # 2つの曲の間の相対的なストレッチ比率
        ratio = ts_rate_j / ts_rate_i
        
        # ratio倍テンポが速いとき、対数軸上では右（プラス方向）に何マスずれるべきか？
        # pythonのfloatと演算するとDouble型（float64）に昇格してしまうため、float()で戻す
        shift_logs = (torch.log(ratio) / self.log_bin_width).float()
        
        # モデルの出力を1つのスカラーに圧縮して比較
        phi_i = self._projection_phi(prob_i)
        phi_j = self._projection_phi(prob_j)
        
        # 理想の理論値: jの分布は「iの分布を shift_logs マス分シフトさせたもの」であるはずなので、
        # phi(j) は phi(i) の (alpha^shift_logs) 倍になるべきである。
        # ※ビューを合わせるために squeeze しています
        shift_k = shift_logs.squeeze(-1)
        
        # PythonのfloatをPytorchの32bitテンソルとして扱う
        alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32, device=self.device)
        target_phi_j = (alpha_tensor ** shift_k) * phi_i
        
        # Huber Loss（Smooth L1 Loss）でペナルティを計算
        # （大きな誤差にはL1、小さな誤差にはL2を適用する強健なロス関数）
        loss = F.huber_loss(phi_j, target_phi_j, delta=1.0)
        
        return loss
