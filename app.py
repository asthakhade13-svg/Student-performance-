from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__, static_folder='static')

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REGISTRY_PATH = os.path.join(MODELS_DIR, 'registry.json')
CSV_PATH = os.path.join(BASE_DIR, 'student_data.csv')

# Ensure models dir exists
os.makedirs(MODELS_DIR, exist_ok=True)

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


# ── MLOps Registry & Validation Helpers ───────────────────────────────

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_version": None, "history": []}


def save_registry(registry):
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)


def validate_student_data(data, is_predict=False):
    try:
        sh = float(data.get('study_hours'))
        att = float(data.get('attendance'))
        pm = float(data.get('previous_marks'))
        ac = float(data.get('assignments_completed'))
        
        if not (0 <= sh <= 12):
            return False, "Study Hours must be between 0 and 12."
        if not (0 <= att <= 100):
            return False, "Attendance must be between 0 and 100%."
        if not (0 <= pm <= 100):
            return False, "Previous Marks must be between 0 and 100."
        if not (0 <= ac <= 10):
            return False, "Assignments Completed must be between 0 and 10."
            
        if not is_predict:
            fs = float(data.get('final_score'))
            if not (0 <= fs <= 100):
                return False, "Final Score must be between 0 and 100."
                
        return True, None
    except (ValueError, TypeError):
        return False, "Input values must be numeric and not empty."


def train_model(df):
    X = df.drop("final_score", axis=1)
    y = df["final_score"]
    
    # Train-test split (adjust test_size if dataset is too small)
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = round(float(mean_absolute_error(y_test, preds)), 2)
    r2 = round(float(r2_score(y_test, preds)), 2)
    r2 = max(-1.0, r2)  # Floor R2 at -1.0 for UI display clarity
    
    # Registering model
    registry = load_registry()
    next_ver_num = len(registry.get("history", [])) + 1
    version = f"v{next_ver_num}"
    
    model_filename = f"model_{version}.pkl"
    model_filepath = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_filepath)
    
    entry = {
        "version": version,
        "path": model_filepath,
        "r2": r2,
        "mae": mae,
        "data_size": len(df),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    registry["history"].append(entry)
    registry["active_version"] = version
    
    # Prune old files: Keep last 5 versions in files, but keep history in registry.json
    if len(registry["history"]) > 5:
        for run in registry["history"][:-5]:
            old_path = run.get("path")
            if old_path and os.path.exists(old_path) and run["version"] != registry["active_version"]:
                try:
                    os.remove(old_path)
                except Exception:
                    pass
                    
    save_registry(registry)
    return model, mae, r2, X.columns.tolist(), version


def get_model_and_stats():
    df = load_or_create_data()
    registry = load_registry()
    active_ver = registry.get("active_version")
    
    if active_ver:
        model_entry = next((item for item in registry["history"] if item["version"] == active_ver), None)
        if model_entry and os.path.exists(model_entry["path"]):
            try:
                model = joblib.load(model_entry["path"])
                return model, model_entry["mae"], model_entry["r2"], ["study_hours", "attendance", "previous_marks", "assignments_completed"], active_ver
            except Exception:
                pass
                
    # If loading active model fails, trigger training
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
        
        # Validation Check
        is_valid, err_msg = validate_student_data(data, is_predict=True)
        if not is_valid:
            return jsonify({"success": False, "error": err_msg}), 400
            
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        previous_marks = float(data['previous_marks'])
        assignments_completed = float(data['assignments_completed'])

        model, mae, r2, _, active_ver = get_model_and_stats()

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
            "r2": r2,
            "active_version": active_ver
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
        
        # Validation Check
        is_valid, err_msg = validate_student_data(data, is_predict=False)
        if not is_valid:
            return jsonify({"success": False, "error": err_msg}), 400

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
        _, mae, r2, _, active_ver = train_model(df)
        return jsonify({
            "success": True, 
            "message": f"Student added and model {active_ver} retrained!", 
            "total": len(df),
            "active_version": active_ver
        })
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
        _, mae, r2, _, active_ver = train_model(df)
        return jsonify({
            "success": True, 
            "message": f"Student deleted and model {active_ver} retrained!", 
            "total": len(df),
            "active_version": active_ver
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    try:
        model, mae, r2, features, active_ver = get_model_and_stats()
        importance = model.feature_importances_.tolist()
        df = load_or_create_data()
        return jsonify({
            "success": True,
            "features": features,
            "importance": importance,
            "mae": mae,
            "r2": r2,
            "total_students": len(df),
            "active_version": active_ver
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        df = load_or_create_data()
        _, mae, r2, _, active_ver = train_model(df)
        return jsonify({
            "success": True, 
            "message": f"Model {active_ver} retrained successfully!", 
            "mae": mae, 
            "r2": r2,
            "active_version": active_ver
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ── MLOps Specific Routes ─────────────────────────────────────────────

@app.route('/api/mlops/history', methods=['GET'])
def get_mlops_history():
    try:
        registry = load_registry()
        # Clean paths for security/brevity before sending
        history_clean = []
        for run in registry.get("history", []):
            run_copy = run.copy()
            if "path" in run_copy:
                run_copy["path"] = os.path.basename(run_copy["path"])
            history_clean.append(run_copy)
            
        return jsonify({
            "success": True,
            "active_version": registry.get("active_version"),
            "history": history_clean
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/mlops/rollback', methods=['POST'])
def rollback_model():
    try:
        data = request.get_json()
        target_version = data.get('version')
        if not target_version:
            return jsonify({"success": False, "error": "No target version provided"}), 400
            
        registry = load_registry()
        # Verify target version exists in history
        model_entry = next((item for item in registry["history"] if item["version"] == target_version), None)
        if not model_entry:
            return jsonify({"success": False, "error": f"Version {target_version} not found in history"}), 400
            
        # Check if the specific checkpoint pickle file still exists
        model_filename = f"model_{target_version}.pkl"
        model_filepath = os.path.join(MODELS_DIR, model_filename)
        
        if not os.path.exists(model_filepath):
            return jsonify({
                "success": False, 
                "error": f"Model checkpoint for {target_version} was pruned or deleted from disk."
            }), 400
            
        registry["active_version"] = target_version
        save_registry(registry)
        
        return jsonify({
            "success": True, 
            "message": f"Successfully rolled back to version {target_version}",
            "active_version": target_version
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/generate-advice', methods=['POST'])
def generate_advice():
    try:
        data = request.get_json()
        study_hours = float(data.get('study_hours', 5.0))
        attendance = float(data.get('attendance', 80.0))
        previous_marks = float(data.get('previous_marks', 70.0))
        assignments_completed = float(data.get('assignments_completed', 6.0))
        predicted_score = float(data.get('predicted_score', 65.0))

        # Check if API Key is configured
        if not GEMINI_API_KEY:
            # Return a high-quality simulated mock recommendation when no key is set
            mock_plan = (
                "### Personalized AI Academic Advisory Report\n\n"
                "> *Notice: Gemini API Key is not set in environment variables. Displaying simulated data-driven plan.*\n\n"
                "#### 1. Strength & Risk Factor Analysis\n"
                f"*   **Attendance ({attendance}%)**: " + 
                ("Excellent! You are attending class regularly, which builds a strong foundation." if attendance >= 85 
                 else "Moderate. Increasing attendance to 85%+ will help capture key topics.") + "\n"
                f"*   **Study Time ({study_hours} hrs/day)**: " + 
                ("Outstanding dedication! You are studying consistently." if study_hours >= 7 
                 else "Room for improvement. Adding 1-2 study hours daily could yield significant score increases.") + "\n"
                f"*   **Assignments ({assignments_completed}/10)**: " + 
                ("High completion! You are practicing and staying on track." if assignments_completed >= 8 
                 else "Critical risk. Completing assignments builds practical skills. Aim for 9/10.") + "\n\n"
                "#### 2. Custom 4-Week Action Planner\n"
                "*   **Week 1 (Establish Foundations)**: Allocate 45 minutes daily to review notes immediately after class. Focus on outstanding assignments.\n"
                "*   **Week 2 (Target Weak Areas)**: Form a study group or attend office hours to address topics where previous marks were dropped.\n"
                "*   **Week 3 (Practice & Reinforce)**: Solve previous practice tests under exam conditions to build time-management confidence.\n"
                "*   **Week 4 (Review & Optimize)**: Focus on light retrieval practice. Get at least 8 hours of sleep before exam day.\n\n"
                "#### 3. Recommended Daily Habits\n"
                "1.  **Pomodoro Study Method**: Work in 25-minute blocks with 5-minute breaks to maintain focus.\n"
                "2.  **Mistake Journaling**: Track incorrect practice answers and solve them from scratch twice.\n"
                "3.  **Active Recall**: Verbally summarize what you learned in class without looking at your slides."
            )
            return jsonify({
                "success": True,
                "advice": mock_plan,
                "is_mock": True
            })

        # Set up Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Construct Prompt
        prompt = (
            "You are EduPredict Advisor, an expert academic counselor. "
            "Analyze the following student profile and generate a highly personalized, structured study advisory report in clean Markdown:\n\n"
            f"- Daily Study Hours: {study_hours} hours/day (target/max: 12)\n"
            f"- Class Attendance: {attendance}%\n"
            f"- Previous Exam Marks: {previous_marks}/100\n"
            f"- Assignments Completed: {assignments_completed}/10\n"
            f"- AI Predicted Final Exam Score: {predicted_score}/100\n\n"
            "Include these exact three sections, using headers:\n"
            "1. Strength & Risk Factor Analysis: Analyze their metrics. Compare parameters and point out major areas causing lower scores vs. areas keeping them afloat.\n"
            "2. Custom 4-Week Action Planner: A specific, week-by-week plan detailing study subjects or practices to raise their grade.\n"
            "3. Recommended Daily Habits: Provide 3-4 specific behavioral habits (e.g. Pomodoro, active recall, sleep guidelines) based on their profile.\n\n"
            "Keep the tone motivational, specific, and actionable. Use bullet points and clean formatting. Do not use generic advice. Do not use any emojis or emoticons in the output."
        )

        response = model.generate_content(prompt)
        advice_text = response.text

        return jsonify({
            "success": True,
            "advice": advice_text,
            "is_mock": False
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    load_or_create_data()
    app.run(debug=False, port=5000)


