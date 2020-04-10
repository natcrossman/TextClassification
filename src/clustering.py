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




def get_new_silhouette_score(labels,k_meansModelTemp, cluster_size):
    #samples > 2 and   cluster_size > 2
    if cluster_size >= 2:
        return silhouette_score(k_meansModelTemp, labels,metric='euclidean')
    return 0

if __name__ == '__main__':
    silhouetteScores = []
    mutualInformationList = []
    silhouetteScoresAgglomerative = []
    mutualInformationListAgglomerative = []
    k_values =  rangeOfTestK_values(2,25,1)
    # pylint: disable=unbalanced-tuple-unpacking
    feature_vectors, targets = load_svmlight_file("training_data_file.TFIDF")
    #get a small better feature set K could be 100 or 1000 or what ever smallerFeatureSet is a csr_matrix
    smallerFeatureSet_matrix= SelectKBest(mutual_info_classif, k=1000).fit_transform(feature_vectors, targets)
    print("Starting clustering")
    for cluster_size in k_values:
        k_meansModelTemp = KMeans(n_clusters=cluster_size)
        k_meansModelTemp.fit(smallerFeatureSet_matrix)
        labels = k_meansModelTemp.labels_
        silhouetteScores.append(get_new_silhouette_score(labels,k_meansModelTemp, cluster_size))
        mutualInformationList.append(normalized_mutual_info_score(targets, labels))
        #Other
        single_linkage_model = AgglomerativeClustering(n_clusters=cluster_size, linkage='ward')
        single_linkage_model.fit(smallerFeatureSet_matrix)
        labels = single_linkage_model.labels_
        silhouetteScoresAgglomerative.append(get_new_silhouette_score(labels,single_linkage_model, cluster_size))
        mutualInformationListAgglomerative.append(normalized_mutual_info_score(targets, labels))
    print("Ending Clustering")


    plt.figure(figsize=(10,7))
    plt.plot(k_values, silhouetteScores)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Silhouette Scores")
    plt.title("KMeans Silhouette Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("SilhouetteScoreKmeans.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


    plt.figure(figsize=(10,7))
    plt.plot(k_values, mutualInformationList)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Normalized Mutual Information Scores")
    plt.title("KMeans Mutual Information Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("mutualInformationListKmeans.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    plt.figure(figsize=(10,7))
    plt.plot(k_values, silhouetteScoresAgglomerative)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Silhouette Scores")
    plt.title("Hierarchical Clustering Silhouette Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("silhouetteScoresAgglomerativeHierarchical.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    plt.figure(figsize=(10,7))
    plt.plot(k_values, mutualInformationListAgglomerative)
    plt.xlabel("Number of K-Values")
    plt.ylabel("Normalized Mutual Information Scores")
    plt.title("Hierarchical Mutual Information Scores", fontsize=22)  
    fig = plt.gcf()
    plt.savefig("mutualInformationListAgglomerativeHierarchical.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

