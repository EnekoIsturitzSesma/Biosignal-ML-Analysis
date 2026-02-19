from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

def train_BCICIV(data, pipeline, cv_strategy, param_grid):
    X = data['X']
    y = data['y']
    subjects = data['subject_ids']  

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, scoring='f1_macro')

    if isinstance(cv_strategy, LeaveOneGroupOut):
        grid_search.fit(X, y, groups=subjects)
    else:
        grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_



