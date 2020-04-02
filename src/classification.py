## @package classification.py
#
#
#
#@copyright     All rights are reserved, this code/project is not Open Source or Free
#@bug           None Documented     
#@author        Nathaniel Crossman & Adam
#
import warnings
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
        self.clf = Classifier()
        self.X = feature_vectors
        self.y = targets
        self.scores = 0.00
    ##
    #   @brief     This method runs the specified Classifier 
    #
    #   @param         self
    #   @return        None
    #   @exception     None
    ##  
    def run(self):
        self.scores = cross_val_score(self.clf, self.X, self.y, cv=5, scoring='f1_macro')
   

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
        print(self.clf.__class__.__name__,": %0.2f (+/- %0.2f)" % (self.scores.mean(), self.scores.std() * 2), "||",dataType)


    ##
    #   @brief     This method returns the  mean scores
    #
    #   @param         self
    #   @return        meanScore
    #   @exception     None
    ##   
    def getMean(self):
        return self.scores.mean()

    ##
    #   @brief     This method returns  95% confidence level
    #
    #   @param         self
    #   @return        Std
    #   @exception     None
    ##  
    def getConfidence_Std(self):
        return (self.scores.std() * 2)  
##
#   @brief     This method launches the classification program  
## 
def run():
    files = ["training_data_file.TF", "training_data_file.IDF","training_data_file.TFIDF"]
    Classifier = [MultinomialNB,BernoulliNB,KNeighborsClassifier,SVC]
    for classifier in Classifier:
        for aFile in files:
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
