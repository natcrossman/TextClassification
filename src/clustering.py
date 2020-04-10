import time
import warnings
from sklearn.metrics import silhouette_score,normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from feature_selection import rangeOfTestK_values
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_selection import chi2, mutual_info_classif
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    silhouetteScores = []
    mutualInformationList = []
    silhouetteScoresAgglomerative = []
    mutualInformationListAgglomerative = []
    k_values =  rangeOfTestK_values(2,26,1)
    # pylint: disable=unbalanced-tuple-unpacking
    feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")


    #Using the results/ experiments done in feature selection, we decided to use the Best K-value to reduce feature set
    #In order to get the best K value we ran several experiments in feature selection to do this. We got the maximum result.
    #Results of experiments:
    # mutual_info_classif - MultinomialNB the best k-value  12800 and  Score:  0.8357452379706963
    # When k-values range was [300 - 17800] = 8 k-values
    # Next, mutual_info_classif MultinomialNB the best k-value position 11100 Score:  0.8413667139030705 
    # When k-values range was [300 - 19800] = 65 k-values
    # For chi2 - MultinomialNB the best k-value position 5500 Score:  0.8869621504029693
    # When k-values range was [300 - 19800] = 65 k-values
    #Based on these experimental results we decided to use the chi2 Method With K value sent to 5500
    smallerFeatureSet_matrix= (SelectKBest(chi2, k=5500).fit_transform(feature_vectors, targets)).toarray()
    print("Starting clustering for both KMean and Hierarchical")
    start_time = time.time()
    for cluster_size in k_values:
        k_meansModelTemp = KMeans(n_clusters=cluster_size)
        k_meansModelTemp.fit(smallerFeatureSet_matrix)
        labels = k_meansModelTemp.labels_
        silhouetteScores.append(silhouette_score(smallerFeatureSet_matrix, labels,metric='euclidean'))
        mutualInformationList.append(normalized_mutual_info_score(targets, labels))
        #Other
        single_linkage_model = AgglomerativeClustering(n_clusters=cluster_size, linkage='ward')
        single_linkage_model.fit(smallerFeatureSet_matrix)
        labels = single_linkage_model.labels_
        silhouetteScoresAgglomerative.append(silhouette_score(smallerFeatureSet_matrix, labels,metric='euclidean'))
        mutualInformationListAgglomerative.append(normalized_mutual_info_score(targets, labels))
    print("Ending Clustering")
    print("Total system timne: \t--- %s seconds ---" % (time.time() - start_time))

    plt.figure(figsize=(10,7))   
    plt.plot(k_values, silhouetteScores,lw=2.5, label = "KMeans")
    plt.plot(k_values, silhouetteScoresAgglomerative, lw=2.5, label = "Hierarchical Clustering")
    plt.xlabel("Number of K-Values")
    plt.ylabel("Silhouette Scores")
    plt.title("KMeans and Hierarchical Clustering", fontsize=22)  
    plt.legend(loc="best", frameon=False)
    fig = plt.gcf()
    plt.savefig("SilhouetteScoreKmeansAndHierarchical.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    plt.figure(figsize=(10,7))   
    plt.plot(k_values, mutualInformationList,lw=2.5, label = "KMeans")
    plt.plot(k_values, mutualInformationListAgglomerative, lw=2.5, label = "Hierarchical Clustering")
    plt.xlabel("Number of K-Values")
    plt.ylabel("Normalized Mutual Information Scores")
    plt.title("KMeans and Hierarchical Clustering", fontsize=22)  
    plt.legend(loc="best",  frameon=False)
    fig = plt.gcf()
    plt.savefig("mutualInformationListKmeansAndHierarchical.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

