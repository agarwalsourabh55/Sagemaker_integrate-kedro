from kedro.pipeline import Pipeline, node

from .nodes import (
    evaluate_model,
    split_data,
    train_model_sagemaker,
    untar_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["master_table", "parameters"],
                outputs=["X_train@pickle", "X_test", "y_train", "y_test"],
            ),
            node(
                func=train_model_sagemaker,
                inputs=["X_train@path", "params:sklearn_estimator_kwargs"],
                outputs="model_path",
            ),
            node(untar_model, inputs="model_path", outputs="regressor"),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
            ),
        ]
    )