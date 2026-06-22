from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__, static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'student_performance_model.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'student_data.csv')

DEFAULT_DATA = {
    "study_hours": [2, 4, 6, 8, 1, 5, 7, 3, 9, 4],
    "attendance": [60, 75, 85, 90, 50, 80, 95, 65, 98, 70],
    "previous_marks": [50, 65, 78, 88, 40, 70, 92, 55, 95, 60],
    "assignments_completed": [4, 6, 8, 9, 2, 7, 10, 5, 10, 6],
    "final_score": [55, 68, 80, 92, 45, 75, 96, 60, 99, 65]
}


def load_or_create_data():
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(DEFAULT_DATA)
        df.to_csv(CSV_PATH, index=False)
    return pd.read_csv(CSV_PATH)


def train_model(df):
    X = df.drop("final_score", axis=1)
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, preds), 2)
    r2 = round(r2_score(y_test, preds), 2)
    joblib.dump(model, MODEL_PATH)
    return model, mae, r2, X.columns.tolist()


def get_model_and_stats():
    df = load_or_create_data()
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        X = df.drop("final_score", axis=1)
        y = df["final_score"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = model.predict(X_test)
        mae = round(mean_absolute_error(y_test, preds), 2)
        r2 = round(r2_score(y_test, preds), 2)
        return model, mae, r2, X.columns.tolist()
    else:
        return train_model(df)


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        previous_marks = float(data['previous_marks'])
        assignments_completed = float(data['assignments_completed'])

        model, mae, r2, _ = get_model_and_stats()

        student = pd.DataFrame({
            "study_hours": [study_hours],
            "attendance": [attendance],
            "previous_marks": [previous_marks],
            "assignments_completed": [assignments_completed]
        })

        predicted_score = round(float(model.predict(student)[0]), 2)
        predicted_score = max(0, min(100, predicted_score))

        # Determine grade
        if predicted_score >= 90:
            grade, grade_class = "A+", "grade-aplus"
        elif predicted_score >= 80:
            grade, grade_class = "A", "grade-a"
        elif predicted_score >= 70:
            grade, grade_class = "B", "grade-b"
        elif predicted_score >= 60:
            grade, grade_class = "C", "grade-c"
        elif predicted_score >= 50:
            grade, grade_class = "D", "grade-d"
        else:
            grade, grade_class = "F", "grade-f"

        return jsonify({
            "success": True,
            "predicted_score": predicted_score,
            "grade": grade,
            "grade_class": grade_class,
            "mae": mae,
            "r2": r2
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    try:
        df = load_or_create_data()
        records = df.to_dict(orient='records')
        return jsonify({"success": True, "data": records, "total": len(records)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/add-student', methods=['POST'])
def add_student():
    try:
        data = request.get_json()
        df = load_or_create_data()
        new_row = {
            "study_hours": float(data['study_hours']),
            "attendance": float(data['attendance']),
            "previous_marks": float(data['previous_marks']),
            "assignments_completed": float(data['assignments_completed']),
            "final_score": float(data['final_score'])
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        # Retrain model with new data
        train_model(df)
        return jsonify({"success": True, "message": "Student added and model retrained!", "total": len(df)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/delete-student/<int:index>', methods=['DELETE'])
def delete_student(index):
    try:
        df = load_or_create_data()
        if index < 0 or index >= len(df):
            return jsonify({"success": False, "error": "Invalid index"}), 400
        df = df.drop(index=index).reset_index(drop=True)
        df.to_csv(CSV_PATH, index=False)
        train_model(df)
        return jsonify({"success": True, "message": "Student deleted and model retrained!", "total": len(df)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    try:
        model, mae, r2, features = get_model_and_stats()
        importance = model.feature_importances_.tolist()
        df = load_or_create_data()
        return jsonify({
            "success": True,
            "features": features,
            "importance": importance,
            "mae": mae,
            "r2": r2,
            "total_students": len(df)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        df = load_or_create_data()
        _, mae, r2, _ = train_model(df)
        return jsonify({"success": True, "message": "Model retrained successfully!", "mae": mae, "r2": r2})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    load_or_create_data()
    app.run(debug=True, port=5000)
