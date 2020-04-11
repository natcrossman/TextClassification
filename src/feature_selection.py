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
import sys
import time
import decimal
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

##
#   @brief     This method generates a range in floats
#
#   @param         start
#   @param         stop
#   @param         step 
#   @return        list of k-values
#   @exception     None
##
def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)
     
##
#   @brief     This method generates the Graphical Plots of the Classifier results
#
#   @param         kvalue
#   @param         f1scores
#   @param         ylabel_vale 
#   @return        list of k-values
#   @exception     None
##
def plotData(kvalue,f1scores, ylabel_vale):
    kvalue = np.array(kvalue)

    multinomialNB_scores = f1scores["MultinomialNB"]
    bernoullinBF_scores = f1scores["BernoulliNB"]
    Kneighbour_scores = f1scores["KNeighborsClassifier"]
    svc_scores = f1scores["SVC"]
    
    plt.figure(figsize=(10,10))

    plt.ylim(0.0, 1)    
    plt.xlim(0, 20000)   
    
    plt.yticks(fontsize=10)    
    plt.xticks(fontsize=10)  

    for y in float_range(0, 1, "0.1"):    
        plt.plot(range(0, 20000), [y] * len(range(0, 20000)), "--", lw=0.5, color="black", alpha=0.3)  
    
    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")    

    plt.plot(kvalue, multinomialNB_scores,lw=2.5, label = "Multinominal NB")
    plt.plot(kvalue, bernoullinBF_scores, lw=2.5, label = "Bernoulli NB")
    plt.plot(kvalue, Kneighbour_scores, lw=2.5, label = "kNN")
    plt.plot(kvalue, svc_scores, lw=2.5, label = "SVM")
    plt.xlabel("K-Value",fontsize=16)
    plt.ylabel(ylabel_vale, fontsize=16)
    plt.legend(loc="best", title="Classifiers", frameon=False)
    plt.title("Results of Feature Selection", fontsize=22)  
    fig = plt.gcf()
    plt.savefig(ylabel_vale +".png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

##
#   @brief     This method generates the number of K value.  
#               Default configuration is from 300 to 20,000 with intervals of 300
#
#   @param         beginning=300
#   @param         end=20000
#   @param         RangeBetween=300
#   @return        list of k-values
#   @exception     None
##
def rangeOfTestK_values(beginning=100,end=20000,RangeBetween=300):
    return list(np.arange(beginning, end, RangeBetween))

# parallelized running of models
def poolRun(f):
    listOfX = []
    k_valueFirst = []
    f1scores =  defaultdict(list)
    Classifiers = getAllClassifier()
    num_workers = (mp.cpu_count())
    # pylint: disable=unbalanced-tuple-unpacking
    feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")
    print( "Starting SelectKBest with ", f.__name__)
    if f.__name__ == "chi2":
            # if you run this code for mutual_info_classif it take for every 2320.036301612854 seconds
            #it is faster for chi2
            k_valueFirst = rangeOfTestK_values()
            start_time = time.time()
            listOfX = [SelectKBest(f, k=i).fit_transform(feature_vectors, targets) for  i in k_valueFirst]
            print( "Finished SelectKBest \t--- %s seconds ---" % (time.time() - start_time), f.__name__)
    else: #It is a lot faster running mutual_info_classif with pool 722.4064118862152 seconds 
        start_time = time.time()
        #keeping range smaller for mutual_info_classif does not really change the result that much
        #k_valueFirst = rangeOfTestK_values(300,20000,300) #65 test for k-value
        k_valueFirst = rangeOfTestK_values(300,20000,2500) #8 test in range 300 to 20000
        pool = mp.Pool(num_workers)
        listOfX += pool.map(partial(getListOfX, f=f, feature_vectors=feature_vectors, targets=targets), k_valueFirst)
        print( "FinishedSelectKBest \t--- %s seconds ---" % (time.time() - start_time),f.__name__)
    print("k-values used: \n", k_valueFirst)
    for classifier in Classifiers: 
        print( "\nStarting ", classifier.__class__.__name__)
        print("--------------------------------------------------------------------------------------------------")
        start_time = time.time()
        meanlist = []
        classification = AnyClassification(classifier,feature_vectors,targets)
        classification.ReplaceScoringType(["f1_macro"])
        pool = mp.Pool(num_workers)

        #These two are just a little slowwer
        #meanlist += pool.starmap(classification.setNewData, listX_y)
        #meanlist += pool.map(partial(classification.setNewData, targets=targets), listOfX)
        meanlist +=pool.starmap(classification.setNewData, zip(listOfX, repeat(targets)))
        maxScoreIndex = meanlist.index(max(meanlist))
        print(classifier.__class__.__name__, "the best k-value position" , k_valueFirst[maxScoreIndex], "Score: ", meanlist[maxScoreIndex], "\n")
        f1scores[classifier.__class__.__name__] = meanlist
        print("Finished ",classifier.__class__.__name__, "\t--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------------------------------------------")
    #Testing 
    print(f1scores)
    #Plot
    plotData(k_valueFirst,f1scores, ("f1_macro with "+ str(f.__name__)))  

##
#   @brief     This method with an array of k value of into specific junk science
#
#   @param         a the array
#   @param         n the split point
#   @return        mean score
#   @exception     None
##
def split_list(a, n):
    k, m = divmod(len(a), n)
    newList = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(newList)


def getListOfX(i, f, feature_vectors,targets):
    #From testing this way is just a little slower
    #listX_y = [[SelectKBest(f, k=i).fit_transform(feature_vectors, targets),targets] for  i in k_valueFirst]
    listOfX = SelectKBest(f, k=i).fit_transform(feature_vectors, targets)
    return listOfX

if __name__ == '__main__':
    #freeze does not work on linux
    mp.freeze_support()
    typeOfF = [chi2, mutual_info_classif]
    start_time = time.time()
    for f in typeOfF:
        mp.freeze_support()
        poolRun(f)
    print("Total system timne: \t--- %s seconds ---" % (time.time() - start_time))

