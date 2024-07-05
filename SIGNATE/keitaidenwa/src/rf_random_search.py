#ランダムサーチ
import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    #学習データセットの読み込み
    df = pd.read_csv("../input/train.csv")

    #特徴量にはprice_range以外のすべての列を使用
    #インデックス列はない
    X = df.drop(["price_range", "three_g", "id"], axis=1).values
    #目的変数の準備
    y = df.price_range.values

    #モデルの定義
    #ランダムフォレストをn_jobs=-1という設定で利用
    #n_jobs=-1はすべて使うという意味
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    #パラメータの探索範囲
    #辞書もしくはパラメータのリストの辞書
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"]
    }

    #ランダムサーチの初期化
    #estimatorはモデル
    #param_distributionsは対象とするパラメータ
    #評価指数は正答率で、独自の評価指標の定義も可能
    #verboseは大きい値を設定すると、より詳細に出力される
    #cv=5はデータセットを5つに分割するという意味
    #(stratified k-fold公差検証法ではない)
    #n_iterは反復数
    #param_distributionsがパラメータのリストの辞書の場合、
    #非復元ランダムサンプリングを実施
    #param_distributionが分布の場合、復元ランダムサンプリングを実施
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    #モデルを学習し、スコアを表示
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")

    print("Best parameters set: ")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
    