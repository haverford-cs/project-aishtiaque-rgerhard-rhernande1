from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
    )


def run_tune_test(learner, params, X, y):
    '''
        Given a learner, hyperparameter dict, and the dataset,
        finds the best hyperparameter combination on the dataset and
        finds the training and testing scores on the 5 folds made on
        the dataset.
    '''
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)
    i = 1
    test_scores = []
    for train_index, test_index in skf.split(X, y):
        print(f"Fold {i}:")
        i += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GridSearchCV(learner, params, cv=3)
        clf.fit(X_train, y_train)
        print()
        print("[GridSearchCV] Best parameters for learner", end=" ")
        print(f"{learner.__class__.__name__}: ")
        print()
        print(clf.best_params_)
        print()
        print("Train Score:", clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
        print("-"*50)
    return test_scores


def show_foldwise_scores(test_scores):
    '''
        Given a list of scores, print out the score-index pair for each score.
    '''
    print("\nFold, Test Accuracy")

    for i in range(len(test_scores)):
        print(i, test_scores[i])

    print()
