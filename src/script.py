from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from git_root import git_root
from lightgbm import LGBMClassifier, LGBMRegressor
from metalearners import RLearner
from metalearners.grid_search import MetaLearnerGridSearch
from metalearners.utils import simplify_output
from shap import TreeExplainer, summary_plot
from sklearn.linear_model import LogisticRegression

_COACHING_COLOR = "green"
_NO_COACHING_COLOR = "red"
_NEUTRAL_COLOR = "grey"

_SEED = 42

_FIG_SIZE_HIST = (15, 10)
_FONT_SIZE = 16


def step_1():
    df = pd.read_csv(Path(git_root()) / "data" / "learning_mindset.csv")
    outcome_column = "achievement_score"
    treatment_column = "intervention"
    feature_columns = [
        column
        for column in df.columns
        if column not in [outcome_column, treatment_column]
    ]
    categorical_feature_columns = [
        "ethnicity",
        "gender",
        "frst_in_family",  # spellchecker:disable-line
        "school_urbanicity",
        "schoolid",
    ]
    # Note that explicitly setting the dtype of these features to category
    # allows both lightgbm as well as shap plots to
    # 1. Operate on features which are not of type int, bool or float
    # 2. Correctly interpret categoricals with int values to be
    #    interpreted as categoricals, as compared to ordinals/numericals.
    for categorical_feature_column in categorical_feature_columns:
        df[categorical_feature_column] = df[categorical_feature_column].astype(
            "category"
        )

    with open("excerpt.md", "w") as txt:
        sample = df.sample(n=5, random_state=_SEED)[
            feature_columns + [treatment_column, outcome_column]
        ]
        txt.write(sample.to_markdown())

    return (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    )


def step_2(df, outcome_column, treatment_column):
    figsize = (_FIG_SIZE_HIST[0] * 2, _FIG_SIZE_HIST[1])
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    axs[0].hist(df[outcome_column], bins=30, color=_NEUTRAL_COLOR)
    axs[0].set_xlabel("outcome")

    axs[1].hist(
        df[df[treatment_column] == 1][outcome_column],
        bins=30,
        density=True,
        alpha=0.5,
        color=_COACHING_COLOR,
        label="coaching",
    )
    axs[1].hist(
        df[df[treatment_column] == 0][outcome_column],
        bins=30,
        density=True,
        alpha=0.5,
        label="no coaching",
        color=_NO_COACHING_COLOR,
    )
    axs[1].set_xlabel("outcome")
    axs[1].legend()

    fig.suptitle("Histograms of outcomes")
    fig.savefig("hist_outcomes.png")

    print(f"fraction of treatment: {df[treatment_column].mean()}")


def step_3(df, outcome_column, treatment_column, feature_columns):
    rlearner = RLearner(
        nuisance_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=2,
        nuisance_model_params={"verbose": -1},
        propensity_model_params={"verbose": -1},
        treatment_model_params={"verbose": -1},
        random_state=_SEED,
    )

    rlearner.fit(
        X=df[feature_columns],
        y=df[outcome_column],
        w=df[treatment_column],
    )

    cate_estimates_rlearner = simplify_output(
        rlearner.predict(
            X=df[feature_columns],
            is_oos=False,
        )
    )

    fig, ax = plt.subplots(figsize=_FIG_SIZE_HIST)
    ax.hist(cate_estimates_rlearner, bins=30, color=_NEUTRAL_COLOR)
    fig.savefig("hist_cates.png")

    return rlearner


def step_4(df, feature_columns, outcome_column, treatment_column):
    gs = MetaLearnerGridSearch(
        metalearner_factory=RLearner,
        metalearner_params={"is_classification": False, "n_variants": 2},
        base_learner_grid={
            "outcome_model": [LGBMRegressor],
            "propensity_model": [LGBMClassifier],
            "treatment_model": [LGBMRegressor],
        },
        param_grid={
            "outcome_model": {
                "LGBMRegressor": {
                    "n_estimators": [25, 50, 100],
                    "max_depth": [-1, 5],
                    "verbose": [-1],
                }
            },
            "treatment_model": {
                "LGBMRegressor": {
                    "n_estimators": [5, 20, 50],
                    "max_depth": [-1, 3, 5],
                    "verbose": [-1],
                }
            },
            "propensity_model": {
                "LGBMClassifier": {
                    "n_estimators": [5, 20, 50],
                    "max_depth": [-1, 3, 5],
                    "verbose": [-1],
                }
            },
        },
        verbose=10,
        random_state=_SEED,
    )

    from sklearn.model_selection import train_test_split

    X_train, X_validation, y_train, y_validation, w_train, w_validation = (
        train_test_split(
            df[feature_columns],
            df[outcome_column],
            df[treatment_column],
            test_size=0.25,
            random_state=_SEED,
        )
    )
    gs.fit(X_train, y_train, w_train, X_validation, y_validation, w_validation)

    with open("grid_search.md", "w") as txt:
        txt.write(gs.results_.to_markdown())

    best_constellation = gs.results_["test_r_loss_1_vs_0"].idxmin()
    print(best_constellation)
    (
        metalearner_name,
        outcome_model_name,
        max_depth_outcome,
        n_estimators_outcome,
        _,
        propensity_model_name,
        max_depth_propensity,
        n_estimators_propensity,
        _,
        treatment_model_name,
        max_depth_treatment,
        n_estimators_treatment,
        _,
    ) = best_constellation

    tuned_rlearner = RLearner(
        nuisance_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=2,
        nuisance_model_params={
            "verbose": -1,
            "n_estimators": n_estimators_outcome,
            "max_depth": max_depth_outcome,
        },
        propensity_model_params={
            "verbose": -1,
            "n_estimators": n_estimators_propensity,
            "max_depth": max_depth_propensity,
        },
        treatment_model_params={
            "verbose": -1,
            "n_estimators": n_estimators_treatment,
            "max_depth": max_depth_treatment,
        },
        random_state=_SEED,
    )

    tuned_rlearner.fit(
        X=df[feature_columns],
        y=df[outcome_column],
        w=df[treatment_column],
    )

    cate_estimates_tuned_rlearner = simplify_output(
        tuned_rlearner.predict(
            X=df[feature_columns],
            is_oos=False,
        )
    )

    default_rlearner = RLearner(
        nuisance_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=2,
        nuisance_model_params={"verbose": -1},
        propensity_model_params={"verbose": -1},
        treatment_model_params={"verbose": -1},
        random_state=_SEED,
    )

    default_rlearner.fit(
        X=df[feature_columns],
        y=df[outcome_column],
        w=df[treatment_column],
    )

    cate_estimates_default_rlearner = simplify_output(
        default_rlearner.predict(
            X=df[feature_columns],
            is_oos=False,
        )
    )

    fig, ax = plt.subplots(figsize=_FIG_SIZE_HIST)
    ax.hist(
        cate_estimates_default_rlearner,
        bins=30,
        color=_NEUTRAL_COLOR,
        label="tuned",
        density=True,
        alpha=0.5,
    )
    ax.hist(
        cate_estimates_tuned_rlearner, bins=30, label="default", density=True, alpha=0.5
    )
    ax.legend()
    fig.savefig("hist_cates_tuned.png")
    return tuned_rlearner


def step_5(rlearner, df, feature_columns):
    figure = plt.figure(figsize=_FIG_SIZE_HIST)
    rlearner_explainer = rlearner.explainer()

    shap_values_rlearner = rlearner_explainer.shap_values(
        X=df[feature_columns], shap_explainer_factory=TreeExplainer
    )
    summary_plot(shap_values_rlearner[0], features=df[feature_columns], show=False)
    figure.tight_layout()
    figure.savefig("shap.png")


def _policy_value(policy, treatment, outcome, propensity_scores):
    n = len(policy)
    policy_value = (
        ((policy == treatment) * outcome) / propensity_scores[np.arange(0, n), policy]
    ).mean()
    return policy_value


def _policy_from_budget(cate_estimates, budget):
    if budget == 0:
        return np.zeros(len(cate_estimates), dtype=int)
    budget_indices = cate_estimates.argsort()[-budget:][::-1]
    policy = np.zeros(len(cate_estimates), dtype=int)
    policy[budget_indices] = 1
    # only if they are positive
    policy = policy * (cate_estimates > 0)
    return policy


def step_6(rlearner: RLearner, df, feature_columns, treatment_column, outcome_column):
    propensity_scores = rlearner.predict_nuisance(
        X=df[feature_columns], model_kind="propensity_model", model_ord=0, is_oos=False
    )

    cate_estimates_rlearner = simplify_output(
        rlearner.predict(
            X=df[feature_columns],
            is_oos=False,
        )
    )

    p_treated = np.linspace(0, 1, 101)
    policy_values = []
    for p in p_treated:
        policy = _policy_from_budget(cate_estimates_rlearner, int(p * len(df)))
        policy_value = _policy_value(
            policy=policy,
            treatment=df[treatment_column],
            outcome=df[outcome_column],
            propensity_scores=propensity_scores,
        )
        policy_values.append(policy_value)

    policy_value_0 = _policy_value(
        np.zeros(df.shape[0], dtype=int),
        treatment=df[treatment_column],
        outcome=df[outcome_column],
        propensity_scores=propensity_scores,
    )

    policy_value_1 = _policy_value(
        np.ones(df.shape[0], dtype=int),
        treatment=df[treatment_column],
        outcome=df[outcome_column],
        propensity_scores=propensity_scores,
    )

    fig, ax = plt.subplots(figsize=_FIG_SIZE_HIST)
    ax.plot(p_treated * len(df), policy_values, label="Policy value")
    ax.plot([0, len(df)], [policy_value_0, policy_value_1], label="UAR")
    ax.set_xlabel("Maximum number of treated students")
    ax.set_ylabel("Policy value")
    ax.legend()

    # we fix the lim so they don't change when plotting the lines
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    for x in [0.3, 0.5]:
        y = policy_values[int(x * 100)]
        y_uar = policy_value_0 + x * (policy_value_1 - policy_value_0)
        text = f"Number of treated: {round(x*len(df))}\nCATE Policy value: {y:.2f}\nUAR Policy value: {y_uar:.2f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        y_size = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.text(
            x * len(df),
            y + 0.02 * y_size,
            text,
            fontsize=_FONT_SIZE,
            bbox=props,
            ha="right",
            va="bottom",
        )
        ax.plot([x * len(df), x * len(df)], [ax.get_ylim()[0], y], "k--", lw=1)
        ax.plot([ax.get_xlim()[0] * len(df), x * len(df)], [y, y], "k--", lw=1)

    fig.tight_layout()
    fig.savefig("policy_value.png")


def step_overlap(df, treatment_column, feature_columns, categorical_feature_columns):

    model = LogisticRegression()

    X = pd.concat(
        (
            df[
                [
                    column
                    for column in feature_columns
                    if column not in categorical_feature_columns
                ]
            ],
            pd.get_dummies(df[categorical_feature_columns], drop_first=True),
        ),
        axis=1,
    )
    model.fit(X, df[treatment_column])
    propensity_scores = model.predict_proba(X)[:, 1]  # Propensity scores

    fig, ax = plt.subplots(figsize=_FIG_SIZE_HIST)

    ax.hist(propensity_scores[df[treatment_column] == 1], label="Treated", alpha=0.5)
    ax.hist(propensity_scores[df[treatment_column] == 0], label="Control", alpha=0.5)
    ax.set_title("Histogram of propensity estimates per treatment assignment group")
    ax.set_xlabel("$\\hat{p}(W=1|X)$")
    ax.legend()
    fig.tight_layout()

    fig.savefig("overlap.png")


def step_conditional_ignorability():
    # TODO
    pass


def main():
    (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    ) = step_1()

    step_overlap(df, treatment_column, feature_columns, categorical_feature_columns)
    step_conditional_ignorability()

    step_2(df, outcome_column, treatment_column)

    step_3(df, outcome_column, treatment_column, feature_columns)

    rlearner_tuned = step_4(df, feature_columns, outcome_column, treatment_column)

    step_5(rlearner_tuned, df, feature_columns)

    step_6(rlearner_tuned, df, feature_columns, treatment_column, outcome_column)


if __name__ == "__main__":
    main()
