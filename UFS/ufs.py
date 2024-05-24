#単変量特徴量選択
#与えられた目的変数の情報を用いて書く特徴量を評価すること
#例）相互情報量、分散分析、カイ二乗検定など
#selectKBest：上位k個の特徴量を保持
#selectPercentile：ユーザが指定した割合で上位の特徴量を保持
#カイ二乗部検定を使用できるのは、非負のデータに限られる。

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        scilkit-learnの複数の手法に対応した
        単変量特徴量選択のためのラッパークラス
        :param n_features: float型のときはSelectPercentileで
        :それ以外のときはSelectKBestを利用
        :param problem_type: 分類か回帰か
        :param scoring: 単位量特徴量の手法名、文字型列
        """
        #指定された問題の種類に対応している手法
        #自由に拡張できる
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "r_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }
        
        #手法が対応してない場合の例外の発生
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")

        #n_featuresがint型の場合はSelectKBest
        #float型の場合はSelectiPercentileを利用
        #float型の場合もint型に変換
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")

    #fit関数
    def fit(self, X, y):
        return self.selection.fit(X, y)

    #transform関数
    def transform(self, X):
        return self.selectiontransform(X)

    #fit_transform関数
    def fit_transform(self, X, y):
        return self.selection.fit_transform(x,y)