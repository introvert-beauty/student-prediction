import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class predictPipeline:
    def __init__(self):
         pass

    # def predict(self,features):
    #     model_path="artifacts/model.pkl"
    #     preprocessor_path="artifacts/preprocessor.pkl"
    #     model=load_object(file_path=model_path)
    #     preprocessor=load_object(file_path=preprocessor_path)
    #     data_scaled=preprocessor.transform(features)
    #     print(type(features))
    #     preds=model.predict(data_scaled)
    #     return preds


    def predict(self, features):
        try:
         model_path = "artifacts/model.pkl"
         preprocessor_path = "artifacts/preprocessor.pkl"

        #  model = load_object(file_path=model_path)
         model_path = "artifacts/model.pkl"
         model = load_object(file_path=model_path)
         print(f"Loaded model type: {type(model)}")  # Should output the model class
    # Should output: <class 'sklearn.ensemble._forest.RandomForestRegressor'> (or similar)

         preprocessor = load_object(file_path=preprocessor_path)

         print(f"Model type: {type(model)}")  # Check the type of model

         if isinstance(features, pd.DataFrame):
            data = features
         else:
            data = pd.DataFrame(features, columns=[
                "StudentID", "Age", "Gender", "Ethnicity", 
                "ParentalEducation", "StudyTimeWeekly", "Absences",
                "Tutoring", "ParentalSupport", "Extracurricular", 
                "Sports", "Music", "Volunteering", "GradeClass"
            ])

         data_scaled =  preprocessor.transform(data)

         preds = model.predict(data_scaled)
        
         return preds
        # In predictPipeline class
       
    
        except Exception as e:
         raise CustomException(f"Prediction failed: {str(e)}", sys)

   


    
# Age	Gender	Ethnicity	ParentalEducation	StudyTimeWeekly	Absences	Tutoring	ParentalSupport	Extracurricular	Sports	Sports	Volunteering	GPA	GradeClass
class Customdata:
    def __init__(self,
                StudentID:int,
                Age:int,
                Gender:int,
                   Ethnicity:int,
                    ParentalEducation:int,
                     StudyTimeWeekly:float,
                    Absences:int,
                    Tutoring:int,
                     ParentalSupport:int,
                     Extracurricular:int,
                     Sports:int,
                     Music:int,
                     Volunteering:int,
                     GradeClass:float):
                self.StudentID=StudentID
                self.Age=Age
                self.Gender=Gender
                self.Ethnicity=Ethnicity
                self.ParentalEducation=ParentalEducation
                self.StudyTimeWeekly=StudyTimeWeekly
                self.Absences=Absences
                self.Tutoring=Tutoring
                self.ParentalSupport=ParentalSupport
                self.Extracurricular=Extracurricular
                self.Sports=Sports
                self.Music=Music
                self.Volunteering=Volunteering
                self.GradeClass=GradeClass
                
    def get_data_as_dataframe(self):
     try:
        custom_data_input_frame = {
            "StudentID":[self.StudentID],
            "Age": [self.Age],
            "Gender": [self.Gender],
            "Ethnicity": [self.Ethnicity],
            "ParentalEducation": [self.ParentalEducation],
            "StudyTimeWeekly": [self.StudyTimeWeekly],
            "Absences": [self.Absences],
            "Tutoring": [self.Tutoring],
            "ParentalSupport": [self.ParentalSupport],
            "Extracurricular": [self.Extracurricular],
            "Sports": [self.Sports],
            "Music": [self.Music],
            "Volunteering": [self.Volunteering],
            "GradeClass": [self.GradeClass]
        }
        return pd.DataFrame(custom_data_input_frame)
     except Exception as e:
        print(f"Error occurred: {e}")




   