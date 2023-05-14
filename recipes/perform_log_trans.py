def perform_log_trans(X, transform_cols=None):
    """Performs log transformation."""
    import numpy as np
    X_copy = X.copy()
    cols = transform_cols or X_copy.select_dtypes(include="number").columns.tolist()
    X_copy[cols] = X_copy[cols].applymap(lambda x: np.log(abs(x)))
    return X_copy
