#ピアソンの相関係数
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

#回帰問題のデータセットを読み込み
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

#pandas データフレームに変換
df = pd.DataFrame(X, columns=col_names)
#相関係数の高い特徴量を作成
df.loc[:, "MedInc_sqrt"] = df.MedInc.apply(np.sqrt)

#ピアソンの相関行列の表示
pc = df.corr()
print(pc)