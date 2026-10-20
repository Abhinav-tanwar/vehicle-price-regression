# Vehicle Price Regression

Mixed-type price regression with train-only preprocessing and held-out evaluation.

**Author and maintainer:** Abhinav Tanwar.

## Run

Requires Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`.

```sh
python analysis.py
python -m unittest discover -s tests
```

## Methods and limitations

The original car-price dataset was found locally. Car IDs and free-text vehicle names are excluded. Imputation, scaling, and one-hot encoding are refitted inside each cross-validation fold. Ridge strength is selected on training data; the 25% test set is held out. Unknown categories are handled by the encoder. The small sample is an illustrative observational benchmark, with prices interpreted in source units.

## Measured results

- **baseline:** rmse = 7,911; mae = 5,910; r squared = -0.0002499.
- **Ridge:** rmse = 2,279; mae = 1,584; r squared = 0.917.

[Full calculated metrics](results/metrics.json) · [Input provenance](data/README.md) · [Method changes](docs/methodology.md). Open `report.html` for the visual summary.

The clean notebook in `notebooks/` delegates to the same module. Notebook cells were syntax-checked with outputs cleared; native scripts and tests were executed.
