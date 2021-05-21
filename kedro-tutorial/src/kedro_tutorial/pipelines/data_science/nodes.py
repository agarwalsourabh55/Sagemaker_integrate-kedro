# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
import tarfile
from typing import Any, Dict

import fsspec
from sagemaker.sklearn.estimator import SKLearn
from sklearn.linear_model import LinearRegression

# <other node functions>


def train_model_sagemaker(
    X_train_path: str, sklearn_estimator_kwargs: Dict[str, Any]
) -> str:
    """Train the linear regression model on SageMaker.

    Args:
        X_train_path: Full S3 path to `X_train` dataset.
        sklearn_estimator_kwargs: Keyword arguments that will be used
            to instantiate SKLearn estimator.

    Returns:
        Full S3 path to `model.tar.gz` file containing the model artifact.

    """
    sklearn_estimator = SKLearn(**sklearn_estimator_kwargs)

    # we need a path to the directory containing both
    # X_train (feature table) and y_train (target variable)
    inputs_dir = X_train_path.rsplit("/", 1)[0]
    inputs = {"train": inputs_dir}

    # wait=True ensures that the execution is blocked
    # until the job finishes on SageMaker
    sklearn_estimator.fit(inputs=inputs, wait=True)

    training_job = sklearn_estimator.latest_training_job
    job_description = training_job.describe()
    model_path = job_description["ModelArtifacts"]["S3ModelArtifacts"]
    return model_path


def untar_model(model_path: str) -> LinearRegression:
    """Unarchive the linear regression model artifact produced
    by the training job on SageMaker.

        Args:
            model_path: Full S3 path to `model.tar.gz` file containing
                the model artifact.

        Returns:
            Trained model.

    """
    with fsspec.open(model_path) as s3_file, tarfile.open(
        fileobj=s3_file, mode="r:gz"
    ) as tar:
        # we expect to have only one file inside the `model.tar.gz` archive
        filename = tar.getnames()[0]
        model_obj = tar.extractfile(filename)
        return pickle.load(model_obj)

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
