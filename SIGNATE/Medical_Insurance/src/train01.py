#ランダムフォレストを使って予測
#パラメータ：
# criterion: gini, 
# max_depth, 
# n_estimators: 200

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 1. 必要なライブラリのインポート

# 2. データの読み込み
df1 = pd.read_csv("../input/train01.csv")
df2 = pd.read_csv("../input/test01.csv")

# 3. 特徴量とターゲット変数の分離
X = df1.drop(columns=["id", "children", "charges"], axis=1) 
y = df1["charges"]
Xtest = df2.drop(columns=["id", "children"], axis=1)


# ランダムフォレストモデルの定義
rf = RandomForestClassifier(
    n_estimators=200,         # 決定木の数
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=2,
    max_depth=10,             # 各決定木の最大深さ
    criterion='gini'
)

# モデルの学習
rf.fit(X, y)

# テストデータに対する予測
y_pred = rf.predict(Xtest)

# 結果をCSVファイルに保存
submission = pd.DataFrame(
    {'id': df2['id'], "charges": y_pred}
    )
submission.to_csv(
    "../output/predict09.csv", 
    index=False, header=False
    )