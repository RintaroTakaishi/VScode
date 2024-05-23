import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    #学習セットの読み込み
    df = pd.read_csv("CFEC2\\input\\train_folds.csv")

    #インデックスと目的変数との列を除き、特徴量とする
    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    #全ての欠損値を”NONE”で補完
    #合わせて、全ての列を文字列型に変換
    #全て質的変数なので問題がない
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    #引数のfold番号と一致しないデータを学習に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.kfold != fold].reset_index(drop=True)

    #初期化
    ohe = preprocessing.OneHotEncoder()

    #学習用と検証用のデータを結合し、OneHotエンコーダを学習
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])

    #学習用データセットを変換
    x_train = ohe.transform(df_train[features])

    #検証用データセットを変換
    x_valid = ohe.transform(df_valid[features])

    #初期化
    model = linear_model.LogisticRegression()
    #モデルの学習
    model.fit(x_train, df_train.target.values)

    #検証用データセットに対する予測
    #AUCを計算するために、予測値が必要
    #１である予測値を利用

    valid_preds = model.predict_proba(x_valid)[:, 1]

    #AUCを計算
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    #AUCを表示
    print(auc)

if __name__ == "__main__":
    #fold番号が0の分割に対して実行
    #引数を変えるだけで、任意の分割に対して実行できる
    run(0)