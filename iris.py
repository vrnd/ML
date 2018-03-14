from urllib.request import urlretrieve
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(iris, sep=',')

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes

print(df.head())
print (df.shape)
print(df.describe())

## class distribution
print(df.groupby('class').size())

## box and whisker plots
#df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, shareybox
#plt.show()

## histograms
#df.hist()
#plt.show()

## scatter plot matrix
#scatter_matrix(df)
#plt.show()

# Split-out validation dataset
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.5
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
final =[]
#print (models)
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)

	mod = model
	mod.fit(X_train,Y_train)
	predictions = mod.predict(X_validation)
	zing = accuracy_score(Y_validation, predictions)
	final.append((name, zing))

	msg = "%s: 'mean: '%f, 'std:' %f, 'acc:' %f" % (name, cv_results.mean(), cv_results.std(), zing)
	print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

##LDA and SVM seem to have the maximum accuracy
#LDA1 = LogisticRegression()
#LDA1.fit(X_train, Y_train)
#predictions = LDA1.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))





