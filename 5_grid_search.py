import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df_1024_clustered = pd.read_csv("data/1024_clustered.csv")
features = ['LSI_all','zeta_all', 'd5_all', 'Sk_all', 'q_all', 'Q6_all']

X = df_1024_clustered[features]
y = df_1024_clustered['labels']

# Split the dataset into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a list of classifiers and their respective hyperparameter grids
classifiers = {
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    'SVM': (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
    }),
    'LogisticRegression': (LogisticRegression(), {
        'C': [0.1, 1, 10]
    }),
    'GradientBoosting': (GradientBoostingClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
    })
}

# Perform the grid search for each classifier and evaluate their accuracy
best_models = {}

for clf_name, (clf, param_grid) in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    best_models[clf_name] = (best_model, f1)

# Find the classifier with the highest accuracy
best_classifier, best_f1_score = max(best_models.items(), key=lambda x: x[1][1])

# Open a text file in write mode
with open('output/grid_search_output.txt', 'w') as file:
    # Write the formatted string to the file
    file.write(f"The best classifier is {best_classifier} with an f1 score of {best_f1_score}\n")
    file.write(f"Best hyperparameters: {best_models[best_classifier][0].get_params()}\n")
