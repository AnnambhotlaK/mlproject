import os
import random
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
            # Define numerical and categorical columsns for pipeline construction

            '''
            numerical_columns = ['draw_size', 'tourney_date', 'p1_id', 'p1_ht', 'p1_age', 'p2_id', 'p2_ht', 'p2_age', 'best_of']
            numerical_columns += ['p1_ace', 'p1_df', 'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_bpSaved', 'p1_bpFaced']
            numerical_columns += ['p2_ace', 'p2_df', 'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms', 'p2_bpSaved', 'p2_bpFaced']
            numerical_columns += ['p1_rank', 'p1_rank_points', 'p2_rank', 'p2_rank_points']
            
            categorical_columns = ['tourney_id', 'surface', 'tourney_level', 'p1_seed', 'p2_seed', 'round']
            '''

            numerical_columns = ['p1_ace', 'p1_df', 'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_bpFaced']
            numerical_columns += ['p2_ace', 'p2_df', 'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_bpFaced']
            numerical_columns += ['p1_rank_points', 'p2_rank_points']

            categorical_columns = ['p1_seed', 'p2_seed']

            num_pipeline = Pipeline(
                steps=[
                    # Imputer will replace nulls with median of column
                    ("imputer", SimpleImputer(strategy="median")),
                    # Scaler will scale data to unit variance for standardization
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical column pipeline constructed")

            cat_pipeline = Pipeline(
                steps=[
                    # Imputer will replace nulls with most frequent_value
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # OH encoder will replace each categorical with numerics
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    # Scaler will scale data to unit variance for standardization
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical column pipeline constructed")

            # Combine pipelines on preprocessor with ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
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

            # On train and test dfs, drop columns, rename winner/loser, scramble winner/loser
            train_df, test_df = self.drop_columns(train_df), self.drop_columns(test_df)
            train_df, test_df = self.rename_winner_loser(train_df), self.rename_winner_loser(test_df)
            train_df, test_df = self.scramble_winner_loser(train_df), self.scramble_winner_loser(test_df)

            print(train_df.head())

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Model will predict match winner
            target_column_name = "match_winner"
            # define numerical/categorical columns here?

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #print(input_feature_train_arr.shape, input_feature_test_arr.shape)
            # Append array of target_feature as column vector to end of 2d array of input_feature
            #train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            #test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            if not isinstance(input_feature_train_arr, np.ndarray):
                input_feature_train_arr = input_feature_train_arr.toarray()
                input_feature_test_arr = input_feature_test_arr.toarray()

            target_train = target_feature_train_df.to_numpy().reshape(-1, 1)
            target_test = target_feature_test_df.to_numpy().reshape(-1, 1)

            train_arr = np.hstack([input_feature_train_arr, target_train])
            test_arr = np.hstack([input_feature_test_arr, target_test])
            
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
    
    # helper function: drop unnecessary columns
    def drop_columns(self, df):
        # define list of columns to drop based on EDA and model training
        '''
        We have 35 numerical features: ['draw_size', 'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_ht', 'winner_age', 'loser_id', 'loser_seed', 'loser_ht', 'loser_age', 'best_of', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points']
        We have 14 categorical features: ['tourney_id', 'tourney_name', 'surface', 'tourney_level', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ioc', 'loser_entry', 'loser_name', 'loser_hand', 'loser_ioc', 'score', 'round']
        '''
        columns_to_drop = ['draw_size', 'tourney_date', 'match_num', 'winner_id', 'winner_ht', 'winner_age', 'loser_id', 'loser_ht', 'loser_age']        
        columns_to_drop += ['best_of', 'minutes', 'w_SvGms', 'w_bpSaved', 'l_SvGms', 'l_bpSaved', 'winner_rank', 'loser_rank']
        columns_to_drop += ['tourney_id', 'tourney_name', 'surface', 'tourney_level', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ioc', 'loser_entry', 'loser_name', 'loser_hand', 'loser_ioc', 'score', 'round']
        df = df.drop(columns=columns_to_drop)
        return df
    
    # helper function: rename winner/loser columns to player1/player2
    def rename_winner_loser(self, df):
        for col in df.columns:
            if 'winner' in col and col != 'match_winner':
                df.rename(columns={col: col.replace('winner', 'p1')}, inplace=True)
            elif col[0] == 'w' and col != 'winner_rank' and col != 'winner_rank_points':
                df.rename(columns={col: col.replace('w', 'p1')}, inplace=True)
            elif 'loser' in col:
                df.rename(columns={col: col.replace('loser', 'p2')}, inplace=True)
            elif col[0] == 'l' and col != 'loser_rank' and col != 'loser_rank_points':
                df.rename(columns={col: col.replace('l', 'p2')}, inplace=True)
        return df
    
    # helper function: scramble winner/loser columns to remove bias
    def scramble_winner_loser(self, df):
        cols1 = [c for c in df.columns if 'p1' in c]
        cols2 = [c for c in df.columns if 'p2' in c]
        cols2_targ = [c.replace('p1', 'p2') for c in cols1]

        # copy df for scrambling
        copy = df.copy()

        # generate len(df.index) / 2 random indices. for those observations, swap columns and set match_winner to 1
        maskIdx = [random.randint(0, len(df.index) - 1) for i in range((int)(len(df.index) / 2))]
        df.loc[maskIdx, cols1] = copy.loc[maskIdx, cols2_targ].values
        df.loc[maskIdx, cols2_targ] = copy.loc[maskIdx, cols1].values
        df.loc[maskIdx, 'match_winner'] = 1

        return df
    


