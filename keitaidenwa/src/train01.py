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
df1 = pd.read_csv("../input/train.csv")
df2 = pd.read_csv("../input/test.csv")

# 3. 特徴量とターゲット変数の分離
X = df1.drop(columns="price_range") 
y = df1['price_range']



# ランダムフォレストモデルの定義
rf = RandomForestClassifier(criterion='gini', max_depth=11, n_estimators=200, random_state=42)

# モデルの学習
rf.fit(X, y)




# テストデータに対する予測
y_pred = rf.predict(df2)

# 結果をCSVファイルに保存
submission = pd.DataFrame({'id': df2['id'], 'price_range': y_pred})
submission.to_csv("../output/predict01.csv", index=False, header=False)