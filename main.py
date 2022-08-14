from sklearn.preprocessing import LabelEncoder, MinMaxScaler, LabelBinarizer, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('ThoraricSurgery.csv')
# print(df.head(5))
# df.info()

# Categorical features handling
encoder = LabelEncoder()
df['DGN'] = encoder.fit_transform(df['DGN'])
df['PRE6'] = encoder.fit_transform(df['PRE6'])
df['PRE7'] = encoder.fit_transform(df['PRE7'])
df['PRE8'] = encoder.fit_transform(df['PRE8'])
df['PRE9'] = encoder.fit_transform(df['PRE9'])
df['PRE10'] = encoder.fit_transform(df['PRE10'])
df['PRE11'] = encoder.fit_transform(df['PRE11'])
df['PRE14'] = encoder.fit_transform(df['PRE14'])
df['PRE17'] = encoder.fit_transform(df['PRE17'])
df['PRE19'] = encoder.fit_transform(df['PRE19'])
df['PRE25'] = encoder.fit_transform(df['PRE25'])
df['PRE30'] = encoder.fit_transform(df['PRE30'])
df['PRE32'] = encoder.fit_transform(df['PRE32'])
df['Risk1Yr'] = encoder.fit_transform(df['Risk1Yr'])

# Normalization of PRE4
PRE4 = df['PRE4']
PRE4 = PRE4.values.reshape(len(PRE4), 1)
trans = MinMaxScaler()
PRE4 = trans.fit_transform(PRE4)
df['PRE4'] = DataFrame(PRE4)

# Normalization of PRE5
PRE5 = df['PRE5']
PRE5 = PRE5.values.reshape(len(PRE5), 1)
trans = MinMaxScaler()
PRE5 = trans.fit_transform(PRE5)
df['PRE5'] = DataFrame(PRE5)

# print(df.head(5))
# df.info()

# Splitting dataset into train set and test set
X = df.drop('Risk1Yr', axis=1)  # Risk1Yr column removed from features
y = df['Risk1Yr']  # class name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # test_set = 94

# function that used in cross validation
def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())
    print()

# Decision Tree Classifier
print("Decision Tree Classifier:")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
scores1 = cross_val_score(classifier,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
classifier_rmse_scores = np.sqrt(-scores1)
display_scores(classifier_rmse_scores)

# Logistic Regression
print("Logistic Regression:")
logistic_regression = LogisticRegression(solver='lbfgs', max_iter=2000)
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
scores2 = cross_val_score(logistic_regression,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
logistic_regression_rmse_scores = np.sqrt(-scores2)
display_scores(logistic_regression_rmse_scores)

# Random Forests Classifier
print("Random Forests Classifier:")
random_forest = RandomForestClassifier(n_estimators=100,oob_score=True,max_features=5)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
scores3 = cross_val_score(random_forest,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
random_forest_rmse_scores = np.sqrt(-scores3)
display_scores(random_forest_rmse_scores)

# SVM Classifier
print("SVM Classifier:")
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("MSE:", metrics.mean_squared_error(y_test,y_pred))
scores4 = cross_val_score(svclassifier,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
svclassifier_rmse_scores = np.sqrt(-scores4)
display_scores(svclassifier_rmse_scores)

# Fine-tuning with GridSearch
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
print("Best parameters after tuning:")
print(grid.best_params_)
grid_predictions = grid.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, grid_predictions))

# Testing on test set with final model
final_model = grid.best_estimator_
final_pred = final_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, final_pred))
print("Precision: ",metrics.precision_score(y_test, final_pred, average='weighted', labels=np.unique(final_pred)))
print("Reacall: ",metrics.recall_score(y_test, final_pred, average='weighted', labels=np.unique(final_pred)))
print("F1-score: ",metrics.f1_score(y_test, final_pred, average='weighted', labels=np.unique(final_pred)))

# # Graphs
# DGN = df['DGN'].value_counts()
# sns.set(style="darkgrid")
# sns.barplot(DGN.index, DGN.values, alpha=0.9)
# plt.title('Frequency Distribution of DGN')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('DGN', fontsize=12)
# plt.show()
#
#
# sns.scatterplot(x='PRE4', y='Risk1Yr', data=df)
# sns.scatterplot(x='PRE5', y='Risk1Yr', data=df)
# sns.scatterplot(x='AGE', y='Risk1Yr', data=df)
# plt.show()


