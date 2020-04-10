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
    k_values =  rangeOfTestK_values(2,25,1)
    # pylint: disable=unbalanced-tuple-unpacking
    feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")
    #get a small better feature set K could be 100 or 1000 or what ever smallerFeatureSet is a csr_matrix
    smallerFeatureSet_matrix= (SelectKBest(mutual_info_classif, k=1000).fit_transform(feature_vectors, targets)).toarray()
    print("Starting clustering for both KMean and Hierarchical")
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


    plt.figure(figsize=(10,7))   
    plt.xlim(2, 26)   
    plt.plot(k_values, silhouetteScores)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Silhouette Scores")
    plt.title("KMeans Silhouette Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("SilhouetteScoreKmeans.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


    plt.figure(figsize=(10,7))
    plt.xlim(2, 26)   
    plt.plot(k_values, mutualInformationList)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Normalized Mutual Information Scores")
    plt.title("KMeans Mutual Information Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("mutualInformationListKmeans.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    plt.figure(figsize=(10,7))
    plt.xlim(2, 26)   
    plt.plot(k_values, silhouetteScoresAgglomerative)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Silhouette Scores")
    plt.title("Hierarchical Clustering Silhouette Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("silhouetteScoresAgglomerativeHierarchical.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    plt.figure(figsize=(10,7))
    plt.xlim(2, 26)   
    plt.plot(k_values, mutualInformationListAgglomerative)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Normalized Mutual Information Scores")
    plt.title("Hierarchical Mutual Information Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("mutualInformationListAgglomerativeHierarchical.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

