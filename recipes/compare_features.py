def compare_models(estimator, X, y, scoring, cv=5, random_state=42, features=None):
    """
    Compares different features.
    """

    from datetime import datetime
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import cross_val_score, KFold

    results = {}

    if isinstance(cv, int):
        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )

    if features is None:
        features = X.columns.tolist()

    for feature in features:
        tic = datetime.now()
        X_temp = X.drop(feature, axis=1)
        print(f"Validating {feature}...")
        cv_results = cross_val_score(
            estimator,
            X_temp,
            y,
            cv=cv,
            scoring=scoring
        )
        results[feature] = cv_results
        toc = datetime.now()
        tic_toc = (toc - tic).seconds / 60
        print(f"Validating {name} done in {tic_toc:.2f} minutes!\n")

    summary = pd.DataFrame(results).describe().transpose()
    print(summary[["min", "mean", "max"]].round(2))
    print("\n")

    plt.boxplot(results.values(), labels=results.keys())
    plt.xticks(rotation=90)
    plt.show()

    return summary
