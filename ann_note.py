from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
#from imblearn.under_sampling import TomekLinks


features = pd.read_csv('neiss_20180620.csv')
features.head(5)
features.shape
labels = np.array(features['farm'])
features= features.drop(['farm'], axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

#has nan element!
#in numpy
np.any(np.isnan(features))
np.all(np.isfinite(features))

# under sampling
def plot_pie(y):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct='%1.1f%%')
    ax.axis('equal')
	
print('Information of the original iris data set: \n {}'.format(
    Counter(labels)))
	
plot_pie(labels)
plt.show()	

# remove Tomek links
# tl = TomekLinks(return_indices=True)
# X_resampled, y_resampled, idx_resampled = tl.fit_sample(features, labels)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# idx_samples_removed = np.setdiff1d(np.arange(features.shape[0]),
                                   # idx_resampled)
# idx_class_0 = y_resampled == 0
# plt.scatter(X_resampled[idx_class_0, 0], X_resampled[idx_class_0, 1],
            # alpha=.8, label='Class #0')
# plt.scatter(X_resampled[~idx_class_0, 0], X_resampled[~idx_class_0, 1],
            # alpha=.8, label='Class #1')
# plt.scatter(features[idx_samples_removed, 0], features[idx_samples_removed, 1],
            # alpha=.8, label='Removed samples')

# plt.title('Under-sampling removing Tomek links')
# plt.legend()
# plt.show()

train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.2)#, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

scores=[]
for val in range (8,15): #get tree depth
#for val in range (18,30): #get num of estimator
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2))
    validated=cross_val_score(clf,train_features,train_labels,cv=10,scoring='roc_auc',n_jobs=8)
    scores.append(validated)

df = pd.DataFrame(scores)	
sns.boxplot(data=df.transpose())
plt.xlabel('Num of times')
plt.ylabel('ROC AUC')
plt.title('ROC AUC score as a function of penalty value')
plt.show()