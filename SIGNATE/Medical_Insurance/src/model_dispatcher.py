from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    "rf": RandomForestClassifier(
        #パラメータ
        n_estimators=200,         # 決定木の数
        max_depth=5,             # 各決定木の最大深さ
        criterion='gini'
    ),
    "svm": SVC(
        kernel="linear",
        C=1.0,
        random_state=42
    )


}