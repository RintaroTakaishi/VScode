#ランダムフォレストモデルを使ってグリッドサーチをどのように行うか
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
    X = df.drop(["price_range", "three_g"], axis=1).values
    #目的変数の準備
    y = df.price_range.values

    #モデルの定義
    #ランダムフォレストをn_jobs=-1という設定で利用
    #n_jobs=-1はすべて使うという意味
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    #パラメータの探索範囲
    #辞書もしくはパラメータのリストの辞書
    param_grid = {
        "n_estimators":[100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }

    #グリッドサーチの初期化
    #estimatorはモデル
    #param_gridは対象とするパラメータ
    #評価指数は正答率で、独自の評価指標の定義も可能
    #verboseは大きい値を設定すると、より詳細に出力される
    #cv=5はデータセットを5つに分割するという意味
    #(stratified k-fold公差検証法ではない)
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
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