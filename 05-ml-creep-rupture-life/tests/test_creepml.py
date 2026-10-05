import numpy as np
import pytest
from sklearn.linear_model import Ridge

from creepml.data import COMPOSITION, GROUP, TARGET, generate_superalloy_dataset, life_hours_true
from creepml.features import LarsonMillerBaseline, add_physics_features, feature_columns
from creepml.models import ColumnSelector, LMPTarget, cross_validate, extrapolation_split, metrics


@pytest.fixture(scope="module")
def df():
    return add_physics_features(generate_superalloy_dataset(n_alloys=30, seed=1))


def test_dataset_schema(df):
    for c in COMPOSITION + ["T_C", "stress_MPa", TARGET, GROUP]:
        assert c in df
    assert np.allclose(df[COMPOSITION + ["Ni"]].sum(axis=1), 100, atol=0.01)
    assert df[TARGET].between(0.5, 5.0).all()


def test_true_life_decreases_with_stress_and_temperature(df):
    c = df.iloc[0]
    assert life_hours_true(c, 900, 200) > life_hours_true(c, 900, 400) > life_hours_true(c, 1000, 400)


def test_lmp_baseline_recovers_exact_relation():
    import pandas as pd
    T = np.repeat([800.0, 900.0, 1000.0], 5)
    s = np.tile(np.geomspace(100, 800, 5), 3)
    lmp = 27 - 1.5 * np.log(s / 100)
    y = 1000 * lmp / (T + 273.15) - 20
    d = pd.DataFrame({"T_C": T, "stress_MPa": s})
    b = LarsonMillerBaseline().fit(d, y)
    assert np.allclose(b.predict(d), y) and b.C == pytest.approx(20)


def test_grouped_cv_and_lmp_target(df):
    cols = feature_columns()
    from sklearn.pipeline import make_pipeline
    model = LMPTarget(Ridge(alpha=1.0), cols)
    oof = cross_validate(model, df, df[TARGET].to_numpy(), df[GROUP].to_numpy(), n_splits=3)
    assert metrics(df[TARGET], oof)["R2"] > 0.5
    tr, te = extrapolation_split(df, 950)
    assert (df["T_C"].iloc[tr] <= 950).all() and (df["T_C"].iloc[te] > 950).all()
    assert make_pipeline(ColumnSelector(cols)).fit_transform(df).shape[1] == len(cols)
