import numpy as np
import warnings
import logging
import random

from constants import DIAGNOSTICS_DIRECTORY, MODELS_DIRECTORY, PLOTS_DIRECTORY, TEST_SET_DIRECTORY, \
    SHAP_VALUES_DIRECTORY, INDIVIDUAL_ID, TARGET, MODEL_TRAINING_DICT, CV_SPLITS
from data.data import get_data
from helpers.helpers import create_directories, make_diagnostic_plots, timer
from modeling.modeling import create_custom_train_test_split, create_custom_cv, construct_pipeline, train_model


warnings.filterwarnings('ignore')
np.random.seed(17)
random.seed(17)


@timer
def main():
    """
    Main execution function to train machine learning models to predict pointsWon in Cy Young races.
    """
    create_directories([DIAGNOSTICS_DIRECTORY, MODELS_DIRECTORY, PLOTS_DIRECTORY, TEST_SET_DIRECTORY,
                        SHAP_VALUES_DIRECTORY], parent=DIAGNOSTICS_DIRECTORY)
    df = get_data()
    make_diagnostic_plots(df)
    x_train, y_train, x_test, y_test = create_custom_train_test_split(df, TARGET, 0.2, INDIVIDUAL_ID)
    custom_cv = create_custom_cv(x_train, INDIVIDUAL_ID, CV_SPLITS)
    for key, value in MODEL_TRAINING_DICT.items():
        train_model(x_train, y_train, x_test, y_test, key, construct_pipeline, value[0], value[1], custom_cv, value[2])


if __name__ == "__main__":
    logging.basicConfig(filename='log.txt', filemode='a', format='%(asctime)s, %(message)s', datefmt='%H:%M:%S',
                        level=20)
    main()
