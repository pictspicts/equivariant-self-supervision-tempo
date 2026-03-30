import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys

# TCNモデルの読み込みパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sst.models.tcn import TCN

def visualize_classification():
    device = torch.device('cpu')
    
    # 1. TCNモデルを「分類モード」で初期化
    TEMPO_RANGE = (0, 300)
    num_classes = TEMPO_RANGE[1] - TEMPO_RANGE[0]
    
    print("Initialising TCN model in classification mode...")
    model = TCN(
        tempo_range=TEMPO_RANGE,
        mode='classification', # ← 多クラス分類
        num_filters=16,
        dropout_rate=0.0
    )
    model.eval()
    
    # 2. ダミーのメルスペクトログラムを入力する
    # 形状: (バッチサイズ, チャンネル, メル周波数, 時間ステップ) => 例: (1, 1, 81, 1361)
    print("\nGenerating dummy mel-spectrogram input...")
    dummy_input = torch.randn(1, 1, 81, 1361)
    
    with torch.no_grad():
        # 推論
        logits = model(dummy_input)
        
    print(f"\nModel output shape (logits): {logits.shape}")
    if logits.shape == (1, num_classes):
        print(f"✅ 成功! モデルは想定通り {num_classes} 次元のクラス（多クラス分類）用ロジットを出力しています。")
    else:
        print(f"❌ 失敗: 出力形状が {logits.shape} になっています。")
    
    # Softmax関数を適用して「確率分布」へと変換（合計が1になる）
    probabilities = F.softmax(logits, dim=-1).numpy()[0]
    
    # 3. 確率分布をプロットして保存
    print("\nPlotting probability distribution...")
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(num_classes)
    plt.plot(x_axis, probabilities, label='Predicted Probabilities', color='b')
    plt.fill_between(x_axis, probabilities, alpha=0.3, color='b')
    
    # 未学習モデルのため分布はランダムになります
    plt.title('TCN Multi-Class Output Distribution (Untrained Model)')
    plt.xlabel('Log-Tempo Bin (Class Index 0 to 299)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    output_path = 'classification_output.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ 可視化画像を保存しました: {os.path.abspath(output_path)}\n")

if __name__ == '__main__':
    visualize_classification()
