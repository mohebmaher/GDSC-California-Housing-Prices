def clean_data(data, drop_labs=None, drop_cols=None, drop_na=False, drop_dups=None, masks=None):
    """Cleans data."""
    data_copy = data.copy()
    if drop_labs is not None:
        data_copy = data_copy.drop(drop_labs, axis=0)
    if drop_cols is not None:
        data_copy = data_copy.drop(drop_cols, axis=1)
    if drop_na is True:
        data_copy.dropna(inplace=True)
    if drop_dups is True:
        data_copy.drop_duplicates(inplace=True)
    if masks is not None:
        for mask in masks:
            data_copy = data_copy.query(f"{mask}").copy()
    return data_copy
