import warnings
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot
from sklearn.datasets import load_svmlight_file
from feature_selection import rangeOfTestK_values,
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_selection import chi2, mutual_info_classif
warnings.filterwarnings('ignore')