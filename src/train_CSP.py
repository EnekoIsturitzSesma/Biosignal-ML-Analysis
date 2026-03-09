from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
import numpy as np

def train_CSP(X, y, subjects, pipeline, param_grid):

    loso = LeaveOneGroupOut()
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=loso, scoring='accuracy', n_jobs=1, refit=True, return_train_score=True)

    grid_search.fit(X, y, groups=subjects)

    results = grid_search.cv_results_
    n_splits = loso.get_n_splits(groups=subjects)

    unique_subjets = np.unique(subjects)

    print("\n Results per fold (best hyperparameters):")
    best_idx = grid_search.best_index_

    for i in range(n_splits):
        train_score = results[f'split{i}_train_score'][best_idx]
        test_score = results[f'split{i}_test_score'][best_idx] 

        print(f'Subject {unique_subjets[i]}: Train Acc = {train_score:.4f} | Val Acc = {test_score:.4f}') 

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_



