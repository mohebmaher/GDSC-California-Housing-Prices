def clean_data(data, drop_cols=None, masks=None, drop_na=False):
    """Cleans data."""
    data_copy = data.copy()
    if drop_cols is not None:
        data_copy = data_copy.drop(drop_cols, axis=1)
    if masks is not None:
        for mask in masks:
            data_copy = data_copy.query(f"{mask}").copy()
    if drop_na is True:
        data_copy.dropna(inplace=True)
    return data_copy
