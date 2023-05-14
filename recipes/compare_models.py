def compare_models(X, y, scoring, cv=5, random_state=42, estimators={}, **kwargs):
    """Compares different estimators."""

    from datetime import datetime
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import cross_val_score, KFold

    estimators.update(kwargs)
    results = {}

    if isinstance(cv, int):
        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )

    for name, estimator in estimators.items():
        tic = datetime.now()
        print(f"Validating {name}...")
        cv_results = cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring
        )
        results[name] = cv_results
        toc = datetime.now()
        tic_toc = (toc - tic).seconds / 60
        print(f"Validating {name} done in {tic_toc:.2f} minutes!\n")

    results = pd.DataFrame(results)
    summary = results.describe().transpose()
    print(summary[["min", "mean", "max"]].applymap(lambda x: f"{x:.2%}"))
    print("\n")

    fig, ax = plt.subplots(figsize=(9, 6))
    results.plot.box(rot=90, ax=ax);

    return summary
