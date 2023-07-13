import sys
sys.path.insert(0,'C:/Users/Hp/Desktop/Data Analyst/mlProject/src')
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from exception import CustomException
from logger import logging
import os

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj = os.path.join('artifact',"Preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ["reading_score","writing_score"]
            categorical_columns = ["gender","race_ethnicity","lunch","parental_level_of_education","test_preparation_course"]


            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing data")

            preprocessing_obj = self.get_data_transformer()
            target_column="math_score"
            numerical_columns = ["reading_score","writing_score"]

            input_feature_train = train_df.drop(columns=[target_column],axis=1)
            target_feature_train = train_df[target_column]

            input_feature_test = test_df.drop(columns=[target_column],axis=1)
            target_feature_test = test_df[target_column]

            logging.info("Applying preprocessing on training and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj,
            )

        except Exception as e:
            raise CustomException(e,sys)