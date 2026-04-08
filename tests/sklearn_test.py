import sklearn as sk

print("Exemple de régression linéaire \n ")
reg = sk.linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[3, 3]]))

print("\nExemple de régression Ridge \n ")
reg = sk.linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[1, 1]]))

print("\nExemple de régression Lasso \n ")
reg = sk.linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[1, 1]]))

x = [[0, 0], [1, 1]]
y = [0, 1]

print("\nExemple de classification SVM \n ")
clf = sk.svm.SVC(decision_function_shape='ovo')
clf.fit(x, y)
clf.predict([[2., 2.]])
clf.support_vectors_
clf.support_
clf.n_support_
dec = clf.decision_function([[0.5, 0.5]])
print(dec)

print("\nExemple de classification SVM linéaire \n ")
lin_clf = sk.svm.LinearSVC()
lin_clf.fit(x, y)
dec = lin_clf.decision_function([[1, 1]])
print(dec.shape)
print(dec)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
iris = load_iris()
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(iris.data, iris.target)
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)

print("\nExemple de classification avec Random Forest Classifier \n")
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
rf_clf.fit(iris.data, iris.target)

fleur_test = [[6.0, 3.0, 5.0, 2.0]]
prediction_rf = rf_clf.predict(fleur_test)
print(f"Classe prédite pour {fleur_test} : Classe {prediction_rf[0]}")

print("Importance des caractéristiques :")
for i in range(len(iris['feature_names'])):
    nom = iris['feature_names'][i]
    importance = rf_clf.feature_importances_[i]
    print(f"- {nom} : {importance:.2f}")
