import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object,evaluate_models

from dataclasses import dataclass
@dataclass 
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model(self, train_array, test_array):
        try:
        # Correct splitting
         x_train, y_train = train_array[:, :-1], train_array[:, -1]
         x_test, y_test = test_array[:, :-1], test_array[:, -1]

         models = {
            "RandomForest": RandomForestRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "LinearRegression": LinearRegression(),
            "KNeighbors": KNeighborsRegressor(),
            "SVR": SVR(),
            "AdaBoost": AdaBoostRegressor(),
        }

         params = {
            "RandomForest": {'n_estimators': [100, 200, 300]},
            "DecisionTree": {"criterion": ['squared_error', 'friedman_mse', 'absolute_error'], "max_depth": [1, 2, 4]},
            "GradientBoosting": {"loss": ['squared_error', 'absolute_error', 'huber'], "learning_rate": [10, 0.01, 0.1]},
            "LinearRegression": {},
            "KNeighbors": {'n_neighbors': [5, 6, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            "SVR": {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "gamma": ['scale', 'auto'], "C": [0.1, 0.01]},
            "AdaBoost": {"n_estimators": [100, 200], "learning_rate": [0.1, 0.01]},
        }

         model_report: dict = evaluate_models(x_train, y_train, x_test, y_test, models=models, param=params)

        # Get the best model score
         best_model_score = max(model_report.values())

        # Find the model associated with the best score
         best_model = [model for model, score in model_report.items() if score == best_model_score][0]  # Get the first match

        # Check if the model score is below a threshold
         if best_model_score < 0.6:
            raise CustomException(f"Best model {best_model} has a score of {best_model_score}, which is below the acceptable threshold.", sys)

         logging.info(f"Best model: {best_model} with score {best_model_score}")

        # Save the best model
         final_model = models[best_model]  # Get the actual model object from the models dictionary
      
         save_object(
    file_path=self.model_trainer_config.model_trainer_file_path,
    obj=final_model  # Ensure this saves the model object
)

         
        
        except Exception as e:
         logging.error(f"Error occurred in ModelTrainer: {str(e)}")  # Log the error
         raise CustomException(e, sys)
