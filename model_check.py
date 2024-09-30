import joblib

model_path = 'artifacts/model.pkl'
model = joblib.load(model_path)

print(f"Loaded model: {model}")
print(f"Model type: {type(model)}")
