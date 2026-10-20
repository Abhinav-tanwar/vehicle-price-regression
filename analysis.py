import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from utils import ROOT, save, reg


def run():
    d = pd.read_csv(ROOT / "data/input.csv")
    y = d.pop("price")
    X = d
    cat = list(X.select_dtypes(exclude=np.number))
    num = [c for c in X if c not in cat]
    a, b = train_test_split(np.arange(len(X)), test_size=0.25, random_state=24)
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ]
    )
    pipeline = Pipeline([("prepare", prep), ("model", Ridge())])
    search = GridSearchCV(
        pipeline,
        {"model__alpha": [0.1, 1, 10, 100]},
        cv=KFold(5, shuffle=True, random_state=24),
        scoring="neg_root_mean_squared_error",
    ).fit(X.iloc[a], y.iloc[a])
    return save(
        dict(
            rows=len(X),
            train_rows=len(a),
            test_rows=len(b),
            alpha=search.best_params_["model__alpha"],
            baseline=reg(y.iloc[b], np.repeat(y.iloc[a].mean(), len(b))),
            Ridge=reg(y.iloc[b], search.predict(X.iloc[b])),
            scope="small observational price benchmark; source target units",
        )
    )


if __name__ == "__main__":
    run()
