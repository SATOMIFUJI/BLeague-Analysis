# ヒートマップの作成

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

### 日本語フォントの設定（Windowsの場合）
matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False

### 順位を反転した「順位スコア」を作成
df_corr = df.copy()
max_rank = df_corr["順位"].max()
df_corr["順位スコア"] = max_rank - df_corr["順位"] + 1

### 目的変数・説明変数リスト（順位の代わりに順位スコアを使う）
corr_vars = [
    '営業収入', '入場料収入', 'スポンサー収入', '物販収入',
    'ユース・スクール関連収入', '配分金（賞金除く）', 'その他',
    '営業費用', '試合関連経費', 'トップチーム人件費',
    'トップチーム運営経費', 'グッズ販売原価（関連経費含む）',
    'ユース・スクール関連経費', '販売費および一般管理費',
    '順位スコア', '勝率'
]

### データの欠損値を除外
corr_df = df_corr[corr_vars].dropna()

### 相関行列を計算
corr_matrix = corr_df.corr()

### ヒートマップ描画
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Bリーグ財務データの相関関係ヒートマップ（順位スコア使用）", fontsize=14)
plt.tight_layout()
plt.show()

