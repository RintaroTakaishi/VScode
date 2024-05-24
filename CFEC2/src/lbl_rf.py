#ランダムフォレスト
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    #学習データセットの読み込み
    df = pd.read_csv("../input/train_folds.csv")

    #インデックスと目的変数とfold番号の列を除き、特徴量とする
    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    #全ての欠損値を"NONE"で補完
    #合わせて、全ての列を文字列型に変換
    #全て質的変数なので問題がない
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    #特徴量のラベルエンコーディング
    for col in features:

        #初期化
        lbl = preprocessing.LabelEncoder()
        
        #ラベルエンコーダの学習
        lbl.fit(df[col])

        #データセットの変換
        df.loc[:, col] = lbl.transform(df[col])

    #引数のfold番号と一致しないデータを学習に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #学習用データセットの準備
    x_train = df_train[features].values

    #検証用データセットの準備
    x_valid = df_valid[features].values

    #初期化
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    #モデルの学習
    model.fit(x_train, df_train.target.values)

    #検証用データセットに対する予測
    #AUCを計算するために、予測値が必要
    #1である予測値を利用
    valid_preds = model.predict_proba(x_valid)[:, 1]

    #AUCを計算
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    #AUCを表示
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)