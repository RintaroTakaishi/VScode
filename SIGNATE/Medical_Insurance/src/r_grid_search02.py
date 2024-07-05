import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    # 学習データセットの読み込み
    df = pd.read_csv("../input/train01.csv")

    # 特徴量にはprice_range以外のすべての列を使用し、インデックス列はないものとします
    X = df.drop(["id", "children", "charges"], axis=1).values
    y = df.charges.values

    # SVMモデルを定義
    svm = SVC()

    # ハイパーパラメータの範囲を定義する
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    # グリッドサーチの実行
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)

    # 各ステップの進捗を表示する
    print("Grid search progress:")
    for i, params in enumerate(grid_search.cv_results_['params']):
        print(f"Step {i+1}/{len(grid_search.cv_results_['params'])} - Parameters: {params}")
        print(f"Mean validation score: {grid_search.cv_results_['mean_test_score'][i]:.2f}")
        print()

    # 最良のモデルとパラメータの表示
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

