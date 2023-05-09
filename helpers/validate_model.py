def compare_models(X, y, scoring, cv=5, random_state=42, estimators={}, **kwargs):
    """
    Compare different estimators.
    """

    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score, KFold
    
    estimators.update(kwargs)
    results = []
    if isinstance(cv, int):
        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )
    
    for name, estimator in estimators.items():
        cv_results = cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring
        )
        print(f"For {name}, results are:")
        print(f"  min = {cv_results.min():.2f}")
        print(f"  mean = {cv_results.mean():.2f}")
        print(f"  max = {cv_results.max():.2f}\n")
        results.append(cv_results)
        
    plt.boxplot(results, labels=estimators.keys())
    plt.xticks(rotation=90)
    plt.show()
    return results
        
        
    