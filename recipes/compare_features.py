def compare_features(estimator, X, y, scoring, cv=5, random_state=42, features=None):
    """Compares different features."""

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

    tic = datetime.now()
    print("Validating current performance...")
    curr_perf = cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring
        )
    results["current_performance"] = curr_perf
    toc = datetime.now()
    tic_toc = (toc - tic).seconds / 60
    print(f"Validating current performance done in {tic_toc:.2f} minutes!\n")

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
        print(f"Validating {feature} done in {tic_toc:.2f} minutes!\n")

    results = pd.DataFrame(results)
    summary = results.describe().transpose()
    summary.sort_values("min", ascending=False, inplace=True)
    print(summary[["min", "mean", "max"]].applymap(lambda x: f"{x:.2%}"))
    print("\n")

    fig, ax = plt.subplots(figsize=(9, 6))
    results.plot.box(rot=90, ax=ax);

    return summary
