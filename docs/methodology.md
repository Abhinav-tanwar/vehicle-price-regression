# Methodology and limitations

The original car-price dataset was found locally. Car IDs and free-text vehicle names are excluded. Imputation, scaling, and one-hot encoding are refitted inside each cross-validation fold. Ridge strength is selected on training data; the 25% test set is held out. Unknown categories are handled by the encoder. The small sample is an illustrative observational benchmark, with prices interpreted in source units.
