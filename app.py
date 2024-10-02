from flask import Flask,request,render_template
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import Customdata,predictPipeline

app=Flask(__name__)
application=app

@app.route("/")
def index():
    return render_template("index.html")

# Age	Gender	Ethnicity	ParentalEducation	StudyTimeWeekly	Absences	Tutoring	ParentalSupport	Extracurricular	Sports	Sports	Volunteering	GPA	GradeClass


@app.route("/predictdata",methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data= Customdata(
            StudentID=request.form.get("StudentID"),
            Age=request.form.get("age"),
            Gender=request.form.get("Gender"),
            Ethnicity=request.form.get("Ethnicity"),
            ParentalEducation=request.form.get("ParentalEducation"),
            StudyTimeWeekly=request.form.get("StudyTimeWeekly"),
            Absences=request.form.get("Absences"),
            Tutoring=request.form.get("Tutoring"),
            ParentalSupport=request.form.get("ParentalSupport"),
            Extracurricular=request.form.get("Extracurricular"),
            Sports=request.form.get("Sports"),
            Music=request.form.get("Music"),
            Volunteering=request.form.get("Volunteering"),
            GradeClass=request.form.get("GradeClass"),
              


        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=predictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    


if __name__ == "__main__":
    app.run(debug=True)

