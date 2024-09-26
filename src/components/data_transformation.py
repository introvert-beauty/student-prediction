import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts/preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            categorical=[]
            numerical=[ 'StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GradeClass']
            num_pipeline=Pipeline(
                steps=[
                    ("inputer",SimpleImputer(strategy="median")),
                    ("standardscaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical),
                ("cat_pipelines",cat_pipeline,categorical)

                ]


            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
           
    
    def intiate_data_transformatio(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read the train and test")

            logging.info("take the properocessor object")


            preprocessor_obj=self.get_data_transformer_object()

            target_column=["GPA"]

            input_feature_train_df=train_df.drop(columns=target_column)
            target_feature_train_df=train_df[target_column]
            input_feature_test_df=test_df.drop(columns=target_column)
            target_feature_test_df=test_df[target_column]

            logging.info("apply preprocessor")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
        

            train_arr=np.c_[
              input_feature_train_arr,np.array(target_feature_train_df)
        ]
            test_arr=np.c_[
              input_feature_test_arr,np.array(target_feature_test_df)
        ]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
        
            return(
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )
        except Exception as e:
            raise CustomException(e,sys)
    
       