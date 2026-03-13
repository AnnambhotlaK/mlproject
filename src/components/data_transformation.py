import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        Function responsible for data transformation via pipelines
        '''
        try:
            numerical_columns = [
                                    'p1_id',
                                    'p2_id',
                                    'p1_ace',
                                    'p1_df',
                                    'p1_svpt',
                                    'p1_1stIn',
                                    'p1_1stWon',
                                    'p1_2ndWon',
                                    'p1_SvGms',
                                    'p1_bpFaced',
                                    'p2_ace',
                                    'p2_df',
                                    'p2_svpt',
                                    'p2_1stIn',
                                    'p2_1stWon',
                                    'p2_2ndWon',
                                    'p2_SvGms',
                                    'p2_bpFaced',
                                    'match_winner'
                                ]

            
            num_pipeline = Pipeline(
                steps=[
                    # Imputer will replace nulls with median of column
                    ("imputer", SimpleImputer(strategy="median")),
                    # Scaler will scale data to unit variance for standardization
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical column pipeline constructed")

            '''
            cat_pipeline = Pipeline(
                steps=[
                    # Imputer will replace nulls with most frequent_value
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # OH encoder will replace each categorical with numerics
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical column pipeline constructed")
            '''

            # Combine pipelines on preprocessor with ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    #("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        '''
        Function responsible for starting data transformation
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Finished reading train and test data")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Model will predict match winner
            target_column_name = "match_winner"
            numerical_columns = [
                                    'p1_id',
                                    'p2_id',
                                    'p1_ace',
                                    'p1_df',
                                    'p1_svpt',
                                    'p1_1stIn',
                                    'p1_1stWon',
                                    'p1_2ndWon',
                                    'p1_SvGms',
                                    'p1_bpFaced',
                                    'p2_ace',
                                    'p2_df',
                                    'p2_svpt',
                                    'p2_1stIn',
                                    'p2_1stWon',
                                    'p2_2ndWon',
                                    'p2_SvGms',
                                    'p2_bpFaced',
                                    'match_winner'
                                ]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object.")

            # Save preprocessing_obj as a pkl file in the file path
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)

