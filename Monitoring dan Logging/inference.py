import joblib
import numpy as np

model = joblib.load('../MLProject/credit_score_model.pkl')

sample_data = np.random.rand(1, 23)

prediction = model.predict(sample_data)

print("Hasil prediksi:", prediction)