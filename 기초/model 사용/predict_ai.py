import joblib

model = joblib.load("model.pkl")

# user_level=2, difficulty=3
prediction = model.predict([[2, 3]])

print(prediction)