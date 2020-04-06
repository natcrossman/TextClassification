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
import time
import warnings
import numpy as np
from itertools import repeat
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from collections import defaultdict
from classification import getAllClassifier
from classification import AnyClassification
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2, mutual_info_classif



# def run_mutual_info_classif(k_value):
#     #Note: Since we got better performance and result using TFIDF from previous experinment we are going to use you it
#     fileT = "training_data_file.TFIDF"
#     k_value = rangeOfTestK_values()
#     Classifier = getAllClassifier()
#     f1scores =  defaultdict(list)

#     for classifier in Classifier:
#         for i in k_value:
#             # pylint: disable=unbalanced-tuple-unpacking
#             feature_vectors, targets = load_svmlight_file(aFile)
#             X = SelectKBest(mutual_info_classif, k=i).fit_transform(feature_vectors, targets)
#             classification = AnyClassification(classifier,X ,targets)
#             classification.ReplaceScoringType(["f1_macro"])
#             classification.run()
#             f1scores[classifier].append(classification.getMean())
#     plotData(k_value,f1scores, "f1_macro with mutual_info_classif")  
#    print(kvalue)
    # print(f1scores)
    # print(len(f1scores["MultinomialNB"]))
    # print(f1scores["MultinomialNB"])
    # print(len(kvalue))

    # print(len(f1scores["BernoulliNB"]))
    # print(f1scores["BernoulliNB"])

    # print(len(f1scores["KNeighborsClassifier"]))
    # print(f1scores["KNeighborsClassifier"])
    
    # print(len(f1scores["SVC"]))

    # print(f1scores["SVC"])          

def plotData(kvalue,f1scores, ylabel_vale):
    kvalue = np.array(kvalue)

    multinomialNB_scores = f1scores["MultinomialNB"]
    bernoullinBF_scores = f1scores["BernoulliNB"]
    Kneighbour_scores = f1scores["KNeighborsClassifier"]
    svc_scores = f1scores["SVC"]

    plt.figure(figsize=(9,9))
    plt.plot(kvalue, multinomialNB_scores,label = " Multinominal NB")
    plt.plot(kvalue, bernoullinBF_scores, label = "Bernoulli NB")
    plt.plot(kvalue, Kneighbour_scores, label = "kNN")
    plt.plot(kvalue, svc_scores, label = "SVM")
    plt.xlabel("K")
    plt.ylabel(ylabel_vale)
    plt.legend(loc = 'best')
    plt.show()

def rangeOfTestK_values(beginning=300,end=20000,RangeBetween=300):
    return list(np.arange(beginning, end, RangeBetween))

#Note: Since we got better performance and result using TFIDF from previous experinment we are going to use you it
def run_chi2(k_value,classifier, fileT):
    tempList = []
    for i in k_value:
        # pylint: disable=unbalanced-tuple-unpacking
        feature_vectors, targets = load_svmlight_file(fileT)
        X = SelectKBest(chi2, k=i).fit_transform(feature_vectors, targets)
        classification = AnyClassification(classifier,X ,targets)
        classification.ReplaceScoringType(["f1_macro"])
        classification.run()
        tempList.append(classification.getMean())
    return tempList

def poolRun(func):
    f1scores =  defaultdict(list)
    Classifiers = getAllClassifier()
    k_valueFirst = rangeOfTestK_values()
    num_workers = (mp.cpu_count()- 2)
    # pylint: disable=unbalanced-tuple-unpacking
    feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")

    start_time = time.time()
    listX_y = [[SelectKBest(chi2, k=i).fit_transform(feature_vectors, targets),targets] for  i in k_valueFirst]
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    listOfX = [SelectKBest(chi2, k=i).fit_transform(feature_vectors, targets) for  i in k_valueFirst]
    
    print("--- %s seconds ---" % (time.time() - start_time))


    slipList = split_list(listX_y,num_workers)
    for classifier in Classifiers: 
        start_time = time.time()
        meanlist = []
        classification = AnyClassification(classifier,feature_vectors,targets)
        classification.ReplaceScoringType(["f1_macro"])

        pool = mp.Pool(num_workers)
        #meanlist.append(pool.starmap(classification.setNewData, listX_y))
        meanlist +=pool.starmap(classification.setNewData, zip(listOfX, repeat(targets)))
        #meanlist.append(pool.map(partial(classification.setNewData, targets=targets), listOfX))
        f1scores[classifier.__class__.__name__] = meanlist
        
        print(classifier.__class__.__name__, "--- %s seconds ---" % (time.time() - start_time))



       
       

    #print(f1scores)
    plotData(k_valueFirst,f1scores, "f1_macro with chi2")   

#Splits the K value list into parts
def split_list(a, n):
    k, m = divmod(len(a), n)
    newList = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(newList)


if __name__ == '__main__':
    mp.freeze_support()
    start_time = time.time()
    poolRun(run_chi2)
    print("--- %s seconds ---" % (time.time() - start_time))

