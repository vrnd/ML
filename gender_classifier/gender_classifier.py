from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

#[height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

X_test = [[190,90,45],[134,35,37],[169,70,42],[178, 83,44]]

def best_classifier_score (X, Y, X_test):
	got_pre = []
	classifiers = [DecisionTreeClassifier(), SVC(), GaussianNB(), GradientBoostingClassifier()]
	for classi in classifiers:
		clf = classi
		clf = clf.fit(X,Y)
		prediction = clf.predict(X_test)
		got_pre.append(classi)
		got_pre.append(prediction)
	return got_pre


print(best_classifier_score(X,Y,X_test))

'''clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190,70,43]])

print(prediction)
'''