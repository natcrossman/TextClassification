## @package classification.py
#We will experiment with several algorithmss: 
#Multinominal Naive Bayes, Bernoulli Naive Bayes, k Nearest Neighbor, and SVM. 
#
#To simplify the task, we use the default parameters for alll the classifiers besides SVC(class_weight="balanced")
#
#We used cross validation for the whole dataset to get more reliable estimation of the classifier performance - 
# We looked at the mean and standard deviation for the selected metric. 
#
#We Repot the mean and 2*std of 5-fold with f1_macro, precision_macro, and recall_macro, respectively. 
#@note: a reasonble classifier typically has F1 score in the range >0.5.

#@copyright     All rights are reserved, this code/project is not Open Source or Free
#@bug           None Documented     
#@author        Nathaniel Crossman & Adam
#
import warnings
import itertools
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


##
# @brief     Need to remove warning about f1 
#
# @bug       None documented yet   
#
warnings.filterwarnings('ignore')

##
# @brief     This class used to abstract away which classifier is being run
#
# @bug       None documented yet   
#
class AnyClassification:
    ##
    #    @param         self
    #    @param         docID
    #    @return        None
    #    @brief         The constructor. 
    #    @exception     None documented yet
    ##
    def __init__(self, Classifier,feature_vectors,targets,):
        self.clf = Classifier
        self.X = feature_vectors
        self.y = targets
        self.scores = []
        self.scoring = ['f1_macro','precision_macro','recall_macro']
    ##
    #   @brief     This method runs the specified Classifier 
    #
    #   @param         self
    #   @return        None
    #   @exception     None
    ##  
    def run(self):
        for scoringtpye in self.scoring:
            self.scores.append(cross_val_score(self.clf, self.X, self.y, cv=5, scoring=scoringtpye))
   

    ##
    #   @brief     This method pritns the result of classification
    #
    #   @param         self
    #   @param         dataType Holds which training data set was used
    #   @see           https://github.com/scikit-learn/scikit-learn/issues/1940
    #   @return        None
    #   @exception     None
    ##   
    def printResults(self, dataType):
        #  95% confidence level (scores.std() * 2)
        meanList    = self.getMean()
        stdList     = self.getConfidence_Std()
        typesScorringUseds = self.scoring
    
        print("Classifier:" ,self.clf.__class__.__name__, " running traning data file:" ,dataType)
        print("--------------------------------------------------------------------------------------------------")
        for (mean, std, typeScorringUsed) in zip(meanList, stdList,typesScorringUseds ):
            print(typeScorringUsed, ": %0.2f (+/- %0.2f)" % (mean, std))
        print("--------------------------------------------------------------------------------------------------")

    ##
    #   @brief     This method returns the  mean scores
    #
    #   @param         self
    #   @return        meanScore
    #   @exception     None
    ##   
    def getMean(self):
        meanScorces = []
        for score in self.scores:
            meanScorces.append(score.mean())
        return meanScorces


    ##
    #   @brief     This method returns  95% confidence level
    #
    #   @param         self
    #   @return        Std
    #   @exception     None
    ##  
    def getConfidence_Std(self):
        std = []
        for score in self.scores:
         std.append((score.std() * 2))

        return std   
##
#   @brief     This method launches the classification program  
## 
def run():
    files = ["training_data_file.TF", "training_data_file.IDF","training_data_file.TFIDF"]
    Classifier = [MultinomialNB(),BernoulliNB(),KNeighborsClassifier(),SVC(class_weight="balanced")]
    for classifier in Classifier:
        for aFile in files:
            # pylint: disable=unbalanced-tuple-unpacking
            feature_vectors, targets = load_svmlight_file(aFile)
            classification = AnyClassification(classifier,feature_vectors,targets)
            classification.run()
            classification.printResults(aFile)

if __name__ == '__main__':
    run()


# clf = MultinomialNB()

# feature_vectors, targets = load_svmlight_file("training_data_file.TF")
# scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
# print("MultinomialNB Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# clf2 = BernoulliNB()
# feature_vectors2, targets2 = load_svmlight_file("training_data_file.IDF")
# scores = cross_val_score(clf2, feature_vectors2, targets2, cv=5, scoring='f1_macro')
# print("BernoulliNB Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# clf3 = KNeighborsClassifier()
# feature_vectors3, targets3 = load_svmlight_file("training_data_file.TFIDF")
# scores = cross_val_score(clf3, feature_vectors3, targets3, cv=5, scoring='f1_macro')
# print("KNeighborsClassifier Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# clf4 = SVC()
# feature_vectors4, targets4 = load_svmlight_file("training_data_file.TFIDF")
# scores = cross_val_score(clf4, feature_vectors4, targets4, cv=5, scoring='f1_macro')
# print("SVC Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
