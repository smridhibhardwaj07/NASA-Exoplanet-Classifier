import joblib
import pandas as pd
model = joblib.load("model/random_forest_model.pkl")
print("NASA Exoplanet Classifier")

koi_period = float(input("Enter orbital period: "))
koi_duration = float(input("Enter transit duration: "))
koi_depth = float(input("Enter transit depth: "))
koi_prad = float(input("Enter planet radius: "))
koi_teq = float(input("Enter equilibrium temperature: "))
koi_insol = float(input("Enter insolation flux: "))
koi_model_snr = float(input("Enter model SNR: "))
koi_steff = float(input("Enter star temperature: "))
koi_srad = float(input("Enter star radius: "))


input_data = pd.DataFrame([{
    "koi_period": koi_period,
    "koi_duration": koi_duration,
    "koi_depth": koi_depth,
    "koi_prad": koi_prad,
    "koi_teq": koi_teq,
    "koi_insol": koi_insol,
    "koi_model_snr": koi_model_snr,
    "koi_steff": koi_steff,
    "koi_srad": koi_srad
}])

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)
print("\nPrediction:", prediction[0])
print("\nPrediction probabilities:")

for class_name, prob in zip(model.classes_, probability[0]):
    print(class_name, ":", round(prob * 100, 2), "%")