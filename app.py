from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

try:
    model = joblib.load('waterborne_disease_risk_model.pkl')
    print("Loaded existing model")
except:
    print("No existing model found. Training new model...")
    def train_model(dataset_path="waterborne_disease_dataset.csv"):
        df = pd.read_csv(dataset_path)
        X = df.drop(['primary_disease', 'risk_level', 'risk_score'], axis=1)
        y = df['risk_level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        numeric_features = ['rainfall_mm', 'pH', 'turbidity_NTU', 'dissolved_oxygen_mg_L',
                            'total_coliform_MPN', 'water_temp_C']
        categorical_features = ['state', 'month', 'year']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        model.fit(X_train, y_train)
        joblib.dump(model, 'waterborne_disease_risk_model.pkl')
        return model
    model = train_model('waterborne_disease_dataset.csv')

def estimate_water_quality_and_disease(state, month, year, rainfall=200):
    month_index = MONTHS.index(month)
    is_monsoon = 5 <= month_index <= 8

    ph = np.random.normal(7.2, 0.5)
    turbidity = np.random.normal(4, 2) * (1.3 if is_monsoon else 1.0)
    dissolved_oxygen = np.random.normal(7, 1.5)
    coliform_count = np.random.poisson(100) * (1.5 if is_monsoon else 1.0)
    temperature = np.random.normal(25, 3) + (3 if month_index in [3, 4, 5] else -5 if month_index in [10, 11, 0, 1] else 0)

    if rainfall > 200:
        turbidity += rainfall / 100
        ph -= 0.3
        coliform_count += int(rainfall / 2)

    ph = max(5.5, min(9.0, ph))
    turbidity = max(0.5, turbidity)
    dissolved_oxygen = max(2.0, min(12.0, dissolved_oxygen))
    coliform_count = max(0, coliform_count)
    temperature = max(15, min(35, temperature))

    water_quality = {
        "rainfall_mm": round(rainfall, 2),
        "pH": round(ph, 2),
        "turbidity_NTU": round(turbidity, 2),
        "dissolved_oxygen_mg_L": round(dissolved_oxygen, 2),
        "total_coliform_MPN": int(coliform_count),
        "water_temp_C": round(temperature, 2)
    }

    diseases = ["Cholera", "Typhoid Fever", "Hepatitis A", "Giardiasis", "Dysentery"]
    base_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    probs = base_probs.copy()
    if coliform_count > 500:
        probs[0] += 0.3
        probs[1] += 0.2
    if turbidity > 8:
        probs[4] += 0.2
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]
    selected_diseases = np.random.choice(diseases, size=2, replace=False, p=probs).tolist()

    return water_quality, selected_diseases

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        state = request.form['state']
        month = request.form['month']
        year = int(request.form['year'])
        rainfall = float(request.form['rainfall']) if request.form['rainfall'] else 200

        water_quality, primary_diseases = estimate_water_quality_and_disease(state, month, year, rainfall)

        input_data = pd.DataFrame({
            'state': [state],
            'month': [month],
            'year': [year],
            'rainfall_mm': [rainfall],
            'pH': [water_quality['pH']],
            'turbidity_NTU': [water_quality['turbidity_NTU']],
            'dissolved_oxygen_mg_L': [water_quality['dissolved_oxygen_mg_L']],
            'total_coliform_MPN': [water_quality['total_coliform_MPN']],
            'water_temp_C': [water_quality['water_temp_C']]
        })

        risk_level = model.predict(input_data)[0]
        response = {
            'risk_level': risk_level,
            'water_quality': water_quality,
            'primary_diseases': primary_diseases,
            'state': state,
            'month': month,
            'year': year
        }
        return jsonify(response)

    return render_template('index.html', states=INDIAN_STATES, months=MONTHS)

if __name__ == '__main__':
    app.run(debug=True)