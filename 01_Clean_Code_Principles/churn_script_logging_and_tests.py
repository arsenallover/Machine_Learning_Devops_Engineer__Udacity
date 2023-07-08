'''
this module perform tests of churn_library.py
    - test data import
    - test EDA
    - test data encoder
    - test feature enginering
    - test train model 
    
Author : RP
Date : 8 July 2023
'''
from pathlib import Path
import logging
import pandas as pd
import pytest

from churn_library import import_data, perform_eda, encoder_helper,\
perform_feature_engineering, train_models

DATA_PTH = "./data/bank_data.csv"

CATEGORY_LST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

RESPONSE='Churn'

# Logging config
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# FIXTURES

@pytest.fixture(scope="module", name='path')
def path():
    """
    Fixture - The test function test_import_data() will
    use the return of path() as an argument
    """
    yield DATA_PTH


@pytest.fixture(name='data_frame')
def data_frame():
    """
    Fixture - The test function test_eda() and test_encoder_helper will
    use the return of data_frame() as an argument
    """
    yield import_data(DATA_PTH)


@pytest.fixture(name='encoded_data_frame')
def encoded_data_frame(data_frame):
    """
    Fixture - The test function test_perform_feature_engineering() will
    use the return of encoded_data_frame() as an argument
    """
    yield encoder_helper(data_frame, CATEGORY_LST, RESPONSE)


@pytest.fixture(name='train_test_split')
def train_test_split(encoded_data_frame, response):
    """
    Fixture - The test function train_models() will
    use the return of train_test_split() as an argument
    """
    return perform_feature_engineering(encoded_data_frame, response)


# UNIT TESTS

def test_import_data(path):
    '''
    test import data from csv
    '''

    try:
        data_frame = import_data(path)

    except FileNotFoundError as err:
        logging.error("Testing import_data: File not found")
        raise err

    # Check the df shape
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return data_frame


def test_eda(data_frame):
    """
    Check if EDA results are saved
    """
    perform_eda(data_frame)

    # Check if each file exists
    path = Path("./images/eda")

    for file in [
        "churn_distribution","customer_age_distribution",
                       "marital_status_distribution","total_transaction_distribution",
                       "heatmap"]:
        file_path = path.joinpath(f'{file}.png')
        try:
            assert file_path.is_file()
        except AssertionError as err:
            logging.error("Testing eda: ERROR: EDA results not found.")
            raise err
    logging.info("Testing eda: SUCCESS: EDA results successfully saved!")


def test_encoder_helper(data_frame, category_lst, response):
    '''
    test encoder helper
    '''
    # Check if df is empty
    assert isinstance(data_frame, pd.DataFrame)
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: ERROR: The dataframe doesn't appear to have rows and columns")
        raise err

    data =  encoder_helper(data_frame, category_lst, response)

    # Check if categorical columns exist in df
    try:
        for col in category_lst:
            assert col in data_frame.columns
    except AssertionError as err:
        logging.error("Testing encoder_helper: ERROR: Missing categorical columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS: Categorical columns correctly encoded.")

    return data


def test_perform_feature_engineering(encoded_data_frame, response):
    '''
    test perform_feature_engineering
    '''

    x_train, x_test, y_train, y_test =  perform_feature_engineering(
        encoded_data_frame, response)

    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
    except AssertionError as err:
        logging.error("Testing feature_engineering: ERROR: The shape of train test splits don't match")
        raise err
    logging.info("Testing feature_engineering: SUCCESS: Train test correctly split.")

    return (x_train, x_test, y_train, y_test)


def test_train_models(train_test_split):
    '''
    test train_models
    '''

    x_train, x_test, y_train, y_test = train_test_split

    # Train model
    train_models(x_train, x_test, y_train, y_test)

    # Check if model were saved after done training
    path = Path("./models")

    models = ['logistic_model.pkl', 'rfc_model.pkl']

    for model_name in models:
        model_path = path.joinpath(model_name)
        try:
            assert model_path.is_file()
        except AssertionError as err:
            logging.error("Testing train_models: ERROR: Models not found.")
            raise err
    logging.info("Testing train_models: SUCCESS: Models successfully saved!")


if __name__ == "__main__":
    DF_RAW = test_import_data(DATA_PTH)
    test_eda(DF_RAW)
    ENCODED_DATA = test_encoder_helper(DF_RAW, CATEGORY_LST, RESPONSE)
    FEATURES = test_perform_feature_engineering(ENCODED_DATA, RESPONSE)
    test_train_models(FEATURES)