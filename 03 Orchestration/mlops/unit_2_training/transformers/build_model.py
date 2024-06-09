from typing import Tuple
from pandas import DataFrame, Series
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train_model(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs
) -> Tuple[DictVectorizer, LinearRegression]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    # Prepare the features and target variable
    features_train = df_train[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    target_train = df_train[target]

    # Fit the dict vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(features_train)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, target_train)
    
    print(f"Model intercept: {model.intercept_}")

    return dv, model


@test
def test_model(output, *args) -> None:
    """
    Test the output of the block.
    """
    dv, model = output
    assert dv is not None, 'DictVectorizer is None'
    assert model is not None, 'Model is None'