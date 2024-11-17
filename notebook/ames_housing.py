# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ames Housing

# %%
import lightgbm as lgb
import numpy as np
import optuna
import optunahub
import pandas as pd
import seaborn as sns
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.datasets import fetch_openml
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## データロード

# %%
dataset = fetch_openml(name="house_prices", as_frame=True)

# %%
df_org = dataset["frame"]

# %%
df_org.head()

# %%
df_org.describe()

# %%
df_org.shape

# %%
df_org.isnull().sum()

# %% [markdown]
# オリジナルと同じか確認

# %%
df_from_csv = pd.read_csv("../data/ames-housing.tsv", sep="\t")

# %%
df_from_csv.shape

# %%
df_from_csv.isnull().sum()

# %% [markdown]
# オリジナルのデータを使うことにする

# %% [markdown]
# ### 扱いやすくする

# %%
columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
]

# %%
df = df_from_csv[columns]

# %%
df["Central Air"] = df["Central Air"].map({"N": 0, "Y": 1})

# %%
df.isnull().sum()

# %%
df = df.dropna(axis="index")

# %% [markdown]
# ## 可視化

# %%
sns.pairplot(df)
# %%
# ピアソンの積率相関係数
cm = df.corr(method="pearson")

# %%
cm

# %%
sns.heatmap(cm, annot=True, fmt=".2f")

# %% [markdown]
# ## 前処理

# %%
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# %%
sc_x = StandardScaler()
sc_y = StandardScaler()

# %%
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.to_frame())

# %%
X_temp, X_test, y_temp, y_test = train_test_split(
    X_std, y_std, test_size=0.2, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

# %% [markdown]
# ## 予測

# %% [markdown]
# ハイパーパラメータチューニングなし

# %%
estimator = lgb.LGBMRegressor(verbose=-1)

# %%
scores = cross_val_score(estimator, X_train, y_train, cv=5)

# %%
print(f"{scores = }")
print(f"{np.mean(scores) = }")


# %% [markdown]
# ハイパーパラメータチューニング


# %%
# 以下を変更した場合、sampler の search_space も変更のこと
def objective(trial):
    # params
    params = {
        "objective": "regression",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.02),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50, step=5),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50, step=5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
    }
    # evaluate
    estimator = lgb.LGBMRegressor(**params)
    scores = cross_val_score(estimator, X_train, y_train, cv=5)
    return np.mean(scores)


# %% [markdown]
# default sampler

# %%
study_default = optuna.create_study()
study_default.optimize(objective, n_trials=10)

# %%
best_params_default = study_default.best_params

# %%
print(f"{best_params_default = }")

# %%
best_params_default.update(
    {"objective": "regression", "random_state": 42, "verbose": -1}
)

# %%
estimator_default = lgb.LGBMRegressor(**best_params_default)

# %%
estimator_default.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=20)],
)

# %%
y_pred_default = estimator_default.predict(X_test)

# %%
r2_default = r2_score(y_test, y_pred_default)

# %%
print(f"{r2_default = }")

# %% [markdown]
# CatCMA sampler


# %%
def objective_catcma(trial):
    # params
    params = {
        "objective": "regression",
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
    }
    # evaluate
    estimator = lgb.LGBMRegressor(**params)
    scores = cross_val_score(estimator, X_train, y_train, cv=5)
    return np.mean(scores)


# %%
mod = optunahub.load_module(
    package="samplers/catcma",
)
CatCmaSampler = mod.CatCmaSampler

# %%
study_catcma = optuna.create_study(
    sampler=CatCmaSampler(
        search_space={
            "learning_rate": FloatDistribution(0.01, 0.1),
            "num_leaves": IntDistribution(20, 50),
            "min_data_in_leaf": IntDistribution(10, 50),
            "reg_alpha": FloatDistribution(0.0, 1.0),
            "reg_lambda": FloatDistribution(0.0, 1.0),
        }
    )
)
study_catcma.optimize(objective_catcma, n_trials=10)

# %%
best_params_catcma = study_catcma.best_params

# %%
print(f"{best_params_catcma = }")

# %%
best_params_catcma.update(
    {"objective": "regression", "random_state": 42, "verbose": -1}
)

# %%
estimator_catcma = lgb.LGBMRegressor(**best_params_catcma)

# %%
estimator_catcma.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=20)],
)

# %%
y_pred_catcma = estimator_catcma.predict(X_test)

# %%
r2_catcma = r2_score(y_test, y_pred_catcma)

# %%
print(f"{r2_catcma = }")
