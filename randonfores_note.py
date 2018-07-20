#imbalanced-learn
conda install -c glemaitre imbalanced-learn

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from multiscorer import MultiScorer
#from imblearn.datasets import make_imbalance
#from imblearn.under_sampling import RandomUnderSampler
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

features = pd.read_csv('neiss_20180620.csv')
#features = pd.read_csv('nemsis_20180705_without_time.csv')
#features = pd.read_csv('ntdb_sub_6642.csv')
#features = pd.read_csv('nvss_sub_12012.csv')
features.head(5)
features.shape
# One-hot encode the data using pandas get_dummies
#features = pd.get_dummies(features)
#for neiss
need_dummy=['BDYPT', 'DIAG', 'DISP', 'FMV', 'INTENT', 'STRATUM', 'HISP_C', 'TRMON_C', 'TRDAY_C', 'AGEG4_C', 'AGEG6_C', 'BDYPTG_C', 'RACE2_C', 'RACETH_C', 'INTENT_C', 'PCAUSE_C', 'ICAUSE_C', 'INTCAU_C', 'male', 'agegroup']
for i in need_dummy:
    just_dummies = pd.get_dummies(features[i],i)
    features = pd.concat([features, just_dummies], axis=1)      
    features.drop([i], inplace=True, axis=1)
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
unique, counts = np.unique(features, return_counts=True)
#in pandas
features.isnull().values.any()
features.isnull().any()
features.loc[features['Urbanicity'] != np.float64('nan')]
features.isnull().values.any()
features.isnull().sum()
features=features[np.isfinite(features['Urbanicity'])]
features=features[np.isfinite(features['male'])]
features.dropna(axis=0, how='any', thresh=None, inplace=True)


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.2)#, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#scorer = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro','f1':'f1','roc_auc':'roc+auc'}
scorer = MultiScorer({
    'Accuracy' : (accuracy_score, {}),
    'Precision' : (precision_score, {'pos_label': 1, 'average':'macro'}),
    'Recall' : (recall_score, {'pos_label': 1, 'average':'macro'})})


for val in range (20,60,10): #get penal val
#NESIS 880 
#NTDB 260
#for val in range (3,20): #get tree depth
#for val in range (26,36,1): #get num of estimator
    #clf = RandomForestClassifier(max_depth=7,class_weight={1:880,0:1},max_features='auto',n_estimators=26)
    clf = XGBClassifier(scale_pos_weight=val,max_depth=5,learning_rate =0.001)
    #validated=cross_val_score(clf,train_features,train_labels,cv=5,scoring='f1',n_jobs=8)
    #scores.append(validated)
    cross_val_score(clf,train_features,train_labels,cv=5,scoring=scorer,n_jobs=8)
    results = scorer.get_results()
    for metric_name in results.keys():
        average_score = np.average(results[metric_name])
        print('%s : %f' % (metric_name, average_score))
    print 'time', time.time() - start, '\n\n'

df = pd.DataFrame(scores)	
sns.boxplot(data=df.transpose())
#plt.xlabel('penalties')
#plt.ylabel('ROC AUC')
#plt.title('ROC AUC scores')
plt.show()

clf.fit(train_features, train_labels)
# Print the name and gini importance of each feature
for feature in zip(feature_list, clf.feature_importances_):
    print(feature)
#not working	
#get the direction - partial dependence plot
fig, axs = plot_partial_dependence(clf, X_train, features,feature_names=feature_list),n_jobs=3, grid_resolution=50)
	
#scoring = ['f1']
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.01
sfm = SelectFromModel(clf, threshold=0.001)
# Train the selector
sfm.fit(train_features, train_labels)
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feature_list[feature_list_index])
for feature in zip(feature_list, sfm.feature_importances_):
    print(feature)

#later# Use the forest's predict method on the test data
predictions = clf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#roc
#fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1)
#metrics.auc(fpr, tpr)
print(roc_auc_score(test_labels, predictions))

# Performance sur le train
train_y_pred = clf.predict(train_features)
auc = roc_auc_score(train_labels, train_y_pred)
print("Performance sur le train : ", auc)

# Performance sur le test
test_y_pred = clf.predict(test_features)
auc = roc_auc_score(test_labels, test_y_pred)
print("Performance sur le test : ", auc)


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Print out the feature and importances 
with open('ntdb_reatures.txt','wb') as f:
    for feature in feature_importances:
		f.write(str(feature))
		f.write('\n')
[('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.show()