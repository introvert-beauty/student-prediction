import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_training import ModelTrainer
from src.components.model_training import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts/train.csv")
    test_data_path:str=os.path.join("artifacts/test.csv")
    raw_data_path:str=os.path.join("artifacts/raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("data is reading as dataframe")
            df=pd.read_csv(r"D:\DATA_SCIENCE_PROJECTS\student_prediction_marks\notebooks\Student_performance_data _.csv")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
             
            logging.info("split the data as train and test")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
                # self.data_ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_array,test_array,_=data_transformation.intiate_data_transformatio(train_data,test_data)

    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)


    # model_trainer=ModelTrainer()
    # model_trainer.initiate_model(train_array,test_array)
    

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model(train_array,test_array))

