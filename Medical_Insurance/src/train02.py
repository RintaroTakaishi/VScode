import os
import argparse

import config
import model_dispatcher

import joblib
import pandas as pd
from sklearn import metrics



def run(fold,model):
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

    #model_disparcherを用いてモデルを取り出す
    clf = model_dispatcher.models[model]

    #モデルの学習
    clf.fit(x_train, y_train)

    #検証用データセットに対する予測
    preds = clf.predict(x_valid)

    #正答率を計算し表示
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    #モデルの保存
    #joblib.dump(rf, f"../models/dt_{fold}.bin")
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTOUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    run(
        fold=0, 
        model = args.model
    )
    run(
        fold=1,
        model = args.model
        )
    run(
        fold=2,
        model = args.model
        )
    run(
        fold=3,
        model = args.model)
    run(
        fold=4,
        model = args.model
        )
