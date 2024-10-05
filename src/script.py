from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from git_root import git_root
from lightgbm import LGBMClassifier, LGBMRegressor
from metalearners import RLearner
from metalearners.utils import simplify_output
from shap import TreeExplainer, summary_plot


def step_1():
    df = pd.read_csv(Path(git_root()) / "learning_mindset.csv")
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

    return (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    )


def step_2(df, outcome_column, treatment_column):
    fig, ax = plt.subplots()
    ax.hist(df[outcome_column])
    fig.savefig("hist_outcomes.png")

    print(f"fraction of treatment: {df[treatment_column].mean()}")


def step_3(df, outcome_column, treatment_column, feature_columns):
    rlearner = RLearner(
        nuisance_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=2,
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

    fig, ax = plt.subplots()
    ax.hist(cate_estimates_rlearner)
    fig.savefig("hist_cates.png")

    return rlearner


def step_4():
    # TODO: HPO
    pass


def step_5(rlearner, df, feature_columns):
    figure = plt.figure()
    rlearner_explainer = rlearner.explainer()

    shap_values_rlearner = rlearner_explainer.shap_values(
        X=df[feature_columns], shap_explainer_factory=TreeExplainer
    )
    summary_plot(shap_values_rlearner[0], features=df[feature_columns], show=False)
    figure.tight_layout()
    figure.savefig("shap.png")


def main():
    (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    ) = step_1()

    step_2(df, outcome_column, treatment_column)

    rlearner = step_3(df, outcome_column, treatment_column, feature_columns)

    step_4()

    step_5(rlearner, df, feature_columns)


if __name__ == "__main__":
    main()
