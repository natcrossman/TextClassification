## @package feature_selection.py
#We are going to evaluate two feature selection methods: the chi-squared method and the mutual information method
#And show hpw they performs for this dataset. 
#
#Testing with both methods for a number of Ks and the four classifiers to find out how feature selection 
# performs for this dataset.
# 
# Report with figures of (x-axis: K, y-axis:f1_macro).
#
#@copyright     All rights are reserved, this code/project is not Open Source or Free
#@bug           None Documented     
#@author        Nathaniel Crossman & Adam
#
import warnings
import numpy as np
import matplotlib.pyplot as pyplot
from collections import defaultdict
from classification import getAllClassifier
from classification import AnyClassification
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2, mutual_info_classif






def run_chi2():
    files = ["training_data_file.TF", "training_data_file.IDF","training_data_file.TFIDF"]
    k_value = rangeOfTestK_values()
    Classifier = getAllClassifier()
    f1scores =  defaultdict(list)

    for classifier in Classifier:
        for i in k_value:
            for aFile in files:
                # pylint: disable=unbalanced-tuple-unpacking
                feature_vectors, targets = load_svmlight_file(aFile)
                X = SelectKBest(chi2, k=i).fit_transform(feature_vectors, targets)
                classification = AnyClassification(classifier,X ,targets)
                classification.ReplaceScoringType(["f1_macro"])
                classification.run()
                f1scores[classifier.__class__.__name__].append(classification.getMean())
    plotData(k_value,f1scores, "f1_macro with chi2")   

def run_mutual_info_classif():
    files = ["training_data_file.TF", "training_data_file.IDF","training_data_file.TFIDF"]
    k_value = rangeOfTestK_values()
    Classifier = getAllClassifier()
    f1scores =  defaultdict(list)

    for classifier in Classifier:
        for i in k_value:
            for aFile in files:
                # pylint: disable=unbalanced-tuple-unpacking
                feature_vectors, targets = load_svmlight_file(aFile)
                X = SelectKBest(mutual_info_classif, k=i).fit_transform(feature_vectors, targets)
                classification = AnyClassification(classifier,X ,targets)
                classification.ReplaceScoringType(["f1_macro"])
                classification.run()
                f1scores[classifier].append(classification.getMean())
    plotData(k_value,f1scores, "f1_macro with mutual_info_classif")            

def plotData(kvalue,f1scores, ylabel_vale):
    multinomialNB_scores = f1scores["MultinomialNB"]
    bernoullinBF_scores = ["BernoulliNB"]
    Kneighbour_scores = ["KNeighborsClassifier"]
    svc_scores = ["SVC"]
    pyplot.figure(figsize=(10,10))
    pyplot.plot(kvalue, multinomialNB_scores,label = " Multinominal NB")
    pyplot.plot(kvalue, bernoullinBF_scores, label = "Bernoulli NB")
    pyplot.plot(kvalue, Kneighbour_scores, label = " kNN")
    pyplot.plot(kvalue, svc_scores, label = "SVM")
    pyplot.xlabel("K")
    pyplot.ylabel(ylabel_vale)
    pyplot.legend(loc = 'best')
    pyplot.show()

def rangeOfTestK_values():
    return np.arange(300, 20000, 300)

if __name__ == '__main__':
    run_chi2()
