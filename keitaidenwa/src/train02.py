import os
import config

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def run(fold):
    #学習データセットの読み込み
    df = pd.read_csv(config.TRAINING_FILE)

    #引数のfold番号と一致しないデータを学習に利用
    #合わせてindexをリセット
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #引数のfold番号と一致するデータを学習に利用
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #目的変数の列を削除し、.valuesを用いてnumpy配列に変換
    #目的変数の列はy_trainとして利用
    x_train = df_train.drop("price_range", axis=1).values
    y_train = df_train.price_range.values
    
    #検証用も同様に処理
    x_valid = df_valid.drop("price_range", axis=1).values
    y_valid = df_valid.price_range.values

    #scikit-learnのランダムフォレストのモデル定義
    rf = RandomForestClassifier(
        #パラメータ
    )

    #モデルの学習
    rf.fit(x_train, y_train)

    #検証用データセットに対する予測
    preds = rf.predict(x_valid)

    #正答率を計算し表示
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    #モデルの保存
    #joblib.dump(rf, f"../models/dt_{fold}.bin")
    joblib.dump(
        rf,
        os.path.join(config.MODEL_OUTOUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
