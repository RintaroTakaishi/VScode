import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # データの読み込み
    df1 = pd.read_csv("../input/train01.csv")
    df2 = pd.read_csv("../input/test01.csv")

    # 特徴量とターゲット変数の分離
    X = df1.drop(columns=["id", "children", "charges"], axis=1)
    y = df1["charges"]
    Xtest = df2.drop(columns=["id", "children"], axis=1)

    # トレーニングデータとテストデータに分ける
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVMモデルを定義する
    svm_clf = SVC()

    # ハイパーパラメータの候補を設定する
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    # グリッドサーチを実行する
    grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 最適なハイパーパラメータを表示する
    print(f'Best parameters found: {grid_search.best_params_}')

    # 最適なモデルで予測を行う
    best_svm_clf = grid_search.best_estimator_
    y_pred = best_svm_clf.predict(X_test)

    # モデルの性能を評価する（例：精度）
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # テストデータに対する予測
    y_test_pred = best_svm_clf.predict(Xtest)

    # 結果をCSVファイルに保存
    submission = pd.DataFrame(
        {'id': df2['id'], "charges": y_test_pred}
    )
    submission.to_csv(
        "../output/predict_svm.csv",
        index=False, header=False
    )
