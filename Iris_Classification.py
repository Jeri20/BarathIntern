
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_text

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create and fit the decision tree
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Print the decision tree
tree_rules = export_text(tree, feature_names=iris.feature_names)
print("Decision Tree is:\n", tree_rules)

# Predict the label for a new example
new = [[6.1, 2.8, 4.7, 1.2]]
predicted_label = tree.predict(new)[0]
print("Predicted Label for new example {} is: {}".format(new, predicted_label))
