# ■ヒートマップの作成

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

### 日本語フォントの設定
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

# ■営業収入の重回帰 (全体）

### 欠損値を含まないデータで学習用データを作成 ※予測値として使用する
###  グッズ販売原価　と　順位は多重共線性が高いため削除
train_df = df.dropna()

X_train = train_df[["入場料収入", "スポンサー収入", "物販収入", "試合関連経費","トップチーム人件費", 
                    "トップチーム運営経費","勝率"]]
y_train = train_df["営業収入"]

### 目的変数：営業収入, 説明変数：入場料、スポンサー、物販、トップチーム人件費、試合関連経費、トップチーム運営、
### グッズ販売原価、順位　として
### 回帰分析を行う

### 説明変数を指定
X =  train_df[["入場料収入", "スポンサー収入", "物販収入", "試合関連経費","トップチーム人件費", 
                    "トップチーム運営経費","勝率"]]
y =  train_df["営業収入"]

### 定数項を加える
X = sm.add_constant(X)

### 回帰モデル
model = sm.OLS(y, X).fit()
print(model.summary()) 目的変数：営業収入, 説明変数：入場料、スポンサー、物販、トップチーム人件費、試合関連経費、トップチーム運営、
### グッズ販売原価、順位　として
### 回帰分析を行う

### 説明変数を指定
X =  train_df[["入場料収入", "スポンサー収入", "物販収入", "試合関連経費","トップチーム人件費", 
                    "トップチーム運営経費","勝率"]]
y =  train_df["営業収入"]

### 定数項を加える
X = sm.add_constant(X)

### 回帰モデル
model = sm.OLS(y, X).fit()
print(model.summary())


# ■勝率の単独効果
model4 = smf.ols("営業収入 ~ 勝率", data=df).fit()
print(model4.summary())

# ■勝率 × スポンサー収入の交互作用分析
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

### データ読み込み
df = pd.read_excel("datafiles/卒業レポート/B_R_data1.xlsx")

### 勝率 × 各変数の交互作用列もあらかじめ追加
df["勝率_スポンサー"] = df["勝率"] * df["スポンサー収入"]
df["勝率_入場料"] = df["勝率"] * df["入場料収入"]
df["勝率_物販"] = df["勝率"] * df["物販収入"]

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ■ランダムフォレストの重要度
rf_importance_df = pd.DataFrame({
    "特徴量": ["トップチーム人件費", "スポンサー収入", "入場料収入", "試合関連経費", "その他",
             "物販収入", "グッズ販売原価（関連経費含む）", "販売費および一般管理費",
             "トップチーム運営経費", "その他.1", "ユース・スクール関連収入", "ユース・スクール関連経費",
             "勝率", "順位", "配分金（賞金除く）", "年度", "Unnamed: 0", "チーム名_アルバルク東京"],
    "重要度": [0.4079, 0.3946, 0.0632, 0.0507, 0.0217,
             0.0155, 0.0126, 0.0079,
             0.0040, 0.0039, 0.0038, 0.0025,
             0.0024, 0.0022, 0.0018, 0.0014, 0.0013, 0.0010]
})

### 重回帰の係数（例：順位を除く）
coef_df = pd.DataFrame({
    '特徴量': ["物販収入", "グッズ販売原価（関連経費含む）", "スポンサー収入", "入場料収入",
             "トップチーム運営経費", "試合関連経費", "トップチーム人件費"],
    '係数（重回帰）': [1.522416, 1.029595, 0.962841, 0.729307, 0.360089, 0.311215, 0.088329]
})

### ランダムフォレストから「順位」と「Unnamed: 0」などモデルに関係ない列を除外
exclude_features = ['順位', 'Unnamed: 0', 'その他.1', 'その他', '年度', 'チーム名_アルバルク東京']
rf_filtered = rf_importance_df[~rf_importance_df['特徴量'].isin(exclude_features)]

### 両方の特徴量で共通のものだけ抽出（比較しやすく）
common_features = set(coef_df['特徴量']) & set(rf_filtered['特徴量'])
coef_filtered = coef_df[coef_df['特徴量'].isin(common_features)]
rf_filtered = rf_filtered[rf_filtered['特徴量'].isin(common_features)]

### 両方のデータフレームを特徴量でソートして並びを揃える
coef_filtered = coef_filtered.set_index('特徴量').loc[sorted(common_features)].reset_index()
rf_filtered = rf_filtered.set_index('特徴量').loc[sorted(common_features)].reset_index()

### プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x='係数（重回帰）', y='特徴量', data=coef_filtered, ax=axes[0], palette='Blues_d')
axes[0].set_title('重回帰分析の係数（順位除く）')
axes[0].set_xlabel('係数')
axes[0].set_ylabel('特徴量')

sns.barplot(x='重要度', y='特徴量', data=rf_filtered, ax=axes[1], palette='Greens_d')
axes[1].set_title('ランダムフォレスト特徴量重要度（順位除く）')
axes[1].set_xlabel('重要度')
axes[1].set_ylabel('特徴量')

plt.tight_layout()
plt.show()


# ■将来の営業収入の予測
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df['チーム名'] = df['チーム名'].replace({'栃木ブレックス': '宇都宮ブレックス'})

###
teams = df['チーム名'].unique()

fig, ax = plt.subplots(figsize=(10,6))

for team in teams:
    team_data = df[df['チーム名'] == team].sort_values('年度_年')
    years = team_data['年度_年'].values
    sales = team_data['営業収入'].values

    ### 年度が飛んでいるところで区切る
    split_indices = np.where(np.diff(years) > 1)[0] + 1
    segments = np.split(np.arange(len(years)), split_indices)

    for i, seg in enumerate(segments):
        label = team if i == 0 else None  # 1回だけラベル付け
        ax.plot(years[seg], sales[seg], marker='o', label=label)

ax.set_title("チーム別 営業収入の推移（過去）")
ax.set_xlabel("年度")
ax.set_ylabel("営業収入（千円）")
ax.grid(True)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xticks(sorted(df['年度_年'].unique()))

### 凡例をグラフ外に表示
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)

plt.tight_layout()
plt.show()
