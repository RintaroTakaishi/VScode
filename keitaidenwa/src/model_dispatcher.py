from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    "rf": RandomForestClassifier(
        #パラメータ
    ),
    "svm": SVC(
        kernel="linear",
        C=1.0,
        random_state=42
    )


}