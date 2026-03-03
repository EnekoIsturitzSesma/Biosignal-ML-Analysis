from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

def train_BCICIV(X, y, subjects, pipeline, param_grid):

    loso = LeaveOneGroupOut()
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=loso, scoring='accuracy', n_jobs=1, refit=True)

    grid_search.fit(X, y, groups=subjects)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_



