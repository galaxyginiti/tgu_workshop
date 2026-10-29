"""Преобразование чисел и категорий (на основе 4_features/starter/preprocessing_template.py).

fit — только на train; validation/test — только transform. Списки столбцов
передаются явно (берутся из паспорта признаков), а не «все кроме target».
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FORBIDDEN = {"customer_id", "snapshot_at", "churn_target", "target", "target_end_exclusive", "split", "label_observed"}


def _as_object(values):
    # Полностью пустой категориальный столбец иначе может определиться как float.
    return values.astype(object)


def make_preprocessor(numeric_columns, categorical_columns):
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)
    all_columns = numeric_columns + categorical_columns
    if not all_columns or len(set(all_columns)) != len(all_columns):
        raise ValueError("Provide nonempty, unique, disjoint feature lists")
    if set(all_columns) & FORBIDDEN:
        raise ValueError("Keys, target, split and audit metadata are not model features")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("as_object", FunctionTransformer(_as_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_columns),
        ("categorical", categorical, categorical_columns),
    ], remainder="drop", verbose_feature_names_out=True)
