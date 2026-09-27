import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformaionConfig:
    preprocessor_ob_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformaion:
    def __init__(self):
        self.data_transformaion_config=DataTransformaionConfig()

    def get_data_transformer_object(self):
        logging.info("entered data transformaion")
        try:
            numeric_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 
                                    'parental level of education', 'lunch', 
                                    'test preparation course']

            num_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                         ("scaler",StandardScaler())])
            cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                         ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")),
                                         ("scaler",StandardScaler(with_mean=False))
                                         ])

            logging.info(f"numerical features: {numeric_features}")
            logging.info(f"categorical features: {categorical_features}")

            preprocessor=ColumnTransformer([("num_pipeline",num_pipeline,numeric_features),
                                            ("cat_pipeline", cat_pipeline,categorical_features)])


            return preprocessor

        except Exception as e:
            raise CustomeException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info(f"read train and test data")
            
            logging.info(f"obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math score"
            numeric_features = ['reading score', 'writing score']
            
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]
            logging.info(f"applying preprocessing object on train and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformaion_config.preprocessor_ob_file_path,
                        obj=preprocessing_obj)
            
            return(train_arr,test_arr,self.data_transformaion_config.preprocessor_ob_file_path)

        except Exception as e:
            raise CustomeException(e,sys)
if __name__=="__main__":
    obj=DataTransformaion()
    obj.initiate_data_transformaion()        