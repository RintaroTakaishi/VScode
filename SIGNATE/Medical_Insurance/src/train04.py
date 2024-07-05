import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == "__main__":
    # 2. データの読み込み
    df1 = pd.read_csv("../input/train01.csv")
    df2 = pd.read_csv("../input/test01.csv")

    # 3. 特徴量とターゲット変数の分離
    X = df1.drop(columns=["id", "children", "charges"], axis=1) 
    y = df1["charges"]
    Xtest = df2.drop(columns=["id", "children"], axis=1)

    # トレーニングデータとテストデータに分ける
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ridge分類モデルを定義する
    ridge_clf = RidgeClassifier()

    # ハイパーパラメータの候補を設定する
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000, 4000],
        'random_state': [42],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'tol': [0.001, 0.01, 0.1]
    }

    # グリッドサーチを実行する
    grid_search = GridSearchCV(ridge_clf, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    # 最適なハイパーパラメータを表示する
    print(f'Best parameters found: {grid_search.best_params_}')
    
    # 最適なモデルで予測を行う
    best_ridge_clf = grid_search.best_estimator_
    y_pred = best_ridge_clf.predict(X_test)

    # モデルの性能を評価する（マクロ平均で評価）
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'Precision (Macro): {precision:.4f}')
    print(f'Recall (Macro): {recall:.4f}')
    print(f'F1 Score (Macro): {f1:.4f}')

    # テストデータに対する予測
    y_test_pred = best_ridge_clf.predict(Xtest)

    # 結果をCSVファイルに保存
    submission = pd.DataFrame(
        {'id': df2['id'], "charges": y_test_pred}
        )
    submission.to_csv(
        "../output/predict08.csv", 
        index=False, header=False
        )
