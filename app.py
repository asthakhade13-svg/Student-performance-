from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import google.generativeai as genai
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from models.lstm_model import train_pytorch_model, StudentLSTMRegressor, get_seq_and_static_data

app = Flask(__name__, static_folder='static')

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REGISTRY_PATH = os.path.join(MODELS_DIR, 'registry.json')
CSV_PATH = os.path.join(BASE_DIR, 'student_data.csv')

# Feature Engineering Helpers
FEATURE_COLS = ["attendance", "previous_marks"]
for w in range(1, 5):
    FEATURE_COLS.extend([
        f"study_hours_w{w}",
        f"sleep_hours_w{w}",
        f"lms_logins_w{w}",
        f"assignments_completed_w{w}",
        f"mock_exams_w{w}"
    ])

def add_features(df):
    return df


# Ensure models dir exists
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_default_sequence_data():
    attendance = [60.0, 75.0, 85.0, 90.0, 50.0, 80.0, 95.0, 65.0, 98.0, 70.0, 80.0, 80.0, 85.0, 90.0, 75.0, 60.0, 80.0, 95.0]
    previous_marks = [50.0, 65.0, 78.0, 88.0, 40.0, 70.0, 92.0, 55.0, 95.0, 60.0, 70.0, 70.0, 72.0, 85.0, 60.0, 80.0, 65.0, 50.0]
    final_score = [55.0, 68.0, 80.0, 92.0, 45.0, 75.0, 96.0, 60.0, 99.0, 65.0, 75.0, 75.0, 62.0, 88.0, 73.0, 69.0, 76.0, 64.0]
    
    study_base = [2.0, 4.0, 6.0, 8.0, 1.0, 5.0, 7.0, 3.0, 9.0, 4.0, 5.0, 5.0, 5.0, 6.0, 3.0, 7.0, 4.0, 2.0]
    sleep_base = [7.0, 6.5, 8.0, 7.5, 5.5, 8.0, 7.0, 7.5, 8.5, 6.0, 7.5, 7.5, 4.0, 8.5, 7.5, 5.0, 8.0, 7.5]
    lms_base = [15, 25, 40, 45, 10, 30, 50, 20, 55, 28, 30, 30, 12, 45, 80, 15, 35, 50]
    assign_base = [4, 6, 8, 9, 2, 7, 10, 5, 10, 6, 7, 7, 8, 9, 5, 6, 7, 4]
    mock_base = [48.0, 68.0, 79.0, 87.0, 38.0, 74.0, 91.0, 58.0, 96.0, 62.0, 70.0, 70.0, 55.0, 92.0, 85.0, 62.0, 75.0, 68.0]
    
    data = {
        "attendance": attendance,
        "previous_marks": previous_marks
    }
    
    for w in range(1, 5):
        data[f"study_hours_w{w}"] = [round(max(0.0, min(12.0, b * (0.85 + 0.05 * w))), 1) for b in study_base]
        data[f"sleep_hours_w{w}"] = [round(max(3.0, min(12.0, b * (1.02 - 0.02 * w))), 1) for b in sleep_base]
        data[f"lms_logins_w{w}"] = [int(max(0, b * (0.95 + 0.03 * w))) for b in lms_base]
        data[f"assignments_completed_w{w}"] = [int(max(0, min(10, b * (w / 4.0)))) for b in assign_base]
        data[f"mock_exams_w{w}"] = [round(max(0.0, min(100.0, b * (0.95 + 0.02 * w))), 1) for b in mock_base]
        
    data["final_score"] = final_score
    return data

def load_or_create_data():
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(generate_default_sequence_data())
        df.to_csv(CSV_PATH, index=False)
        return df
    try:
        df = pd.read_csv(CSV_PATH)
        if "study_hours_w1" not in df.columns:
            df = pd.DataFrame(generate_default_sequence_data())
            df.to_csv(CSV_PATH, index=False)
        return df
    except Exception:
        df = pd.DataFrame(generate_default_sequence_data())
        df.to_csv(CSV_PATH, index=False)
        return df


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
        att = float(data.get('attendance'))
        pm = float(data.get('previous_marks'))
        if not (0 <= att <= 100):
            return False, "Attendance must be between 0 and 100%."
        if not (0 <= pm <= 100):
            return False, "Previous Marks must be between 0 and 100."
            
        for w in range(1, 5):
            sh = float(data.get(f'study_hours_w{w}'))
            sl = float(data.get(f'sleep_hours_w{w}'))
            lms = float(data.get(f'lms_logins_w{w}'))
            ac = float(data.get(f'assignments_completed_w{w}'))
            me = float(data.get(f'mock_exams_w{w}'))
            
            if not (0 <= sh <= 12):
                return False, f"Week {w} Study Hours must be between 0 and 12."
            if not (0 <= sl <= 24):
                return False, f"Week {w} Sleep Hours must be between 0 and 24."
            if not (0 <= lms <= 300):
                return False, f"Week {w} LMS Logins must be between 0 and 300."
            if not (0 <= ac <= 10):
                return False, f"Week {w} Assignments Completed must be between 0 and 10."
            if not (0 <= me <= 100):
                return False, f"Week {w} Mock Exam Score must be between 0 and 100."
                
        if not is_predict:
            fs = float(data.get('final_score'))
            if not (0 <= fs <= 100):
                return False, "Final Score must be between 0 and 100."
                
        return True, None
    except (ValueError, TypeError, KeyError) as e:
        return False, f"Input values must be numeric and not empty."


def train_model(df):
    registry = load_registry()
    next_ver_num = len(registry.get("history", [])) + 1
    version = f"v{next_ver_num}"
    
    model_filename = f"model_{version}.pth"
    model_filepath = os.path.join(MODELS_DIR, model_filename)
    
    mae_mean, mae_std, r2_mean = train_pytorch_model(df, model_filepath)
    
    entry = {
        "version": version,
        "path": model_filepath,
        "r2": round(float(r2_mean), 2),
        "mae": round(float(mae_mean), 2),
        "mae_std": round(float(mae_std), 2),
        "data_size": len(df),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    registry["history"].append(entry)
    registry["active_version"] = version
    
    if len(registry["history"]) > 5:
        for run in registry["history"][:-5]:
            old_path = run.get("path")
            if old_path and os.path.exists(old_path) and run["version"] != registry["active_version"]:
                try:
                    os.remove(old_path)
                except Exception:
                    pass
                    
    save_registry(registry)
    return None, mae_mean, r2_mean, FEATURE_COLS, version


def get_model_and_stats():
    df = load_or_create_data()
    registry = load_registry()
    active_ver = registry.get("active_version")
    
    if active_ver:
        model_entry = next((item for item in registry["history"] if item["version"] == active_ver), None)
        if model_entry and os.path.exists(model_entry["path"]):
            try:
                payload = joblib.load(model_entry["path"])
                model = StudentLSTMRegressor()
                model.load_state_dict(payload["model_state"])
                model.eval()
                return model, payload["scaler"], model_entry["mae"], model_entry["r2"], active_ver
            except Exception as e:
                print("Error loading model:", e)
                
    # If loading active model fails, trigger training
    train_model(df)
    return get_model_and_stats()


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
            
        attendance = float(data['attendance'])
        previous_marks = float(data['previous_marks'])
        
        model, scaler, mae, r2, active_ver = get_model_and_stats()
        
        student_data_dict = {
            "attendance": [attendance],
            "previous_marks": [previous_marks]
        }
        for w in range(1, 5):
            student_data_dict[f"study_hours_w{w}"] = [float(data[f"study_hours_w{w}"])]
            student_data_dict[f"sleep_hours_w{w}"] = [float(data[f"sleep_hours_w{w}"])]
            student_data_dict[f"lms_logins_w{w}"] = [float(data[f"lms_logins_w{w}"])]
            student_data_dict[f"assignments_completed_w{w}"] = [float(data[f"assignments_completed_w{w}"])]
            student_data_dict[f"mock_exams_w{w}"] = [float(data[f"mock_exams_w{w}"])]
            
        student_df = pd.DataFrame(student_data_dict)
        student_df = student_df[FEATURE_COLS]
        
        seq_val, _ = get_seq_and_static_data(student_df)
        N_val, T_val, F_val = seq_val.shape
        seq_val_flat = seq_val.reshape(-1, F_val)
        seq_val_scaled = scaler.transform(seq_val_flat).reshape(N_val, T_val, F_val)
        
        with torch.no_grad():
            pred = model(torch.tensor(seq_val_scaled, dtype=torch.float32)).numpy()
            predicted_score = round(float(pred[0][0]), 2)
            predicted_score = max(0.0, min(100.0, predicted_score))
            
        # Calculate SHAP explanations
        import shap
        df = load_or_create_data()
        X_all = df[FEATURE_COLS]
        
        def shap_predict_fn(X_np):
            df_temp = pd.DataFrame(X_np, columns=FEATURE_COLS)
            seq, _ = get_seq_and_static_data(df_temp)
            N_t, T_t, F_t = seq.shape
            seq_flat_t = seq.reshape(-1, F_t)
            seq_scaled = scaler.transform(seq_flat_t).reshape(N_t, T_t, F_t)
            
            with torch.no_grad():
                preds_tensor = model(torch.tensor(seq_scaled, dtype=torch.float32)).numpy()
            return preds_tensor.flatten()
            
        background = X_all
        explainer = shap.KernelExplainer(shap_predict_fn, background)
        shap_vals = explainer.shap_values(student_df, l1_reg=False)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
            
        explanations = []
        for col, val in zip(FEATURE_COLS, shap_vals[0]):
            explanations.append({
                "feature": col,
                "impact": round(float(val), 2)
            })

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
            "active_version": active_ver,
            "explanations": explanations
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
            "attendance": float(data['attendance']),
            "previous_marks": float(data['previous_marks'])
        }
        for w in range(1, 5):
            new_row[f"study_hours_w{w}"] = float(data[f"study_hours_w{w}"])
            new_row[f"sleep_hours_w{w}"] = float(data[f"sleep_hours_w{w}"])
            new_row[f"lms_logins_w{w}"] = float(data[f"lms_logins_w{w}"])
            new_row[f"assignments_completed_w{w}"] = float(data[f"assignments_completed_w{w}"])
            new_row[f"mock_exams_w{w}"] = float(data[f"mock_exams_w{w}"])
        new_row["final_score"] = float(data['final_score'])
        
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
        model, scaler, mae, r2, active_ver = get_model_and_stats()
        
        df = load_or_create_data()
        X_all = df[FEATURE_COLS]
        
        import shap
        def shap_predict_fn(X_np):
            df_temp = pd.DataFrame(X_np, columns=FEATURE_COLS)
            seq, _ = get_seq_and_static_data(df_temp)
            N_t, T_t, F_t = seq.shape
            seq_flat_t = seq.reshape(-1, F_t)
            seq_scaled = scaler.transform(seq_flat_t).reshape(N_t, T_t, F_t)
            with torch.no_grad():
                preds_tensor = model(torch.tensor(seq_scaled, dtype=torch.float32)).numpy()
            return preds_tensor.flatten()
            
        background = X_all
        explainer = shap.KernelExplainer(shap_predict_fn, background)
        sample_subset = X_all
        shap_vals = explainer.shap_values(sample_subset, l1_reg=False)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
            
        importance = np.mean(np.abs(shap_vals), axis=0).tolist()
        
        return jsonify({
            "success": True,
            "features": FEATURE_COLS,
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
            
        # Check if the specific checkpoint file still exists
        model_filename = f"model_{target_version}.pth"
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
        sleep_hours = float(data.get('sleep_hours', 7.5))
        lms_logins = float(data.get('lms_logins', 30.0))
        mock_exams = float(data.get('mock_exams', 70.0))
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
                 else "Critical risk. Completing assignments builds practical skills. Aim for 9/10.") + "\n"
                f"*   **Sleep & Wellness ({sleep_hours} hrs/day)**: " +
                ("Adequate rest. Getting 7-8 hours of sleep protects memory retention." if sleep_hours >= 7
                 else "Sleep deprivation risk. Sleep hours are low; prioritize wellness and sleep for learning retention.") + "\n"
                f"*   **LMS Engagement ({lms_logins} logins/week)**: " +
                ("Active digital participation! Excellent LMS login frequency." if lms_logins >= 25
                 else "Low digital engagement. Try logging in daily to access learning resources.") + "\n"
                f"*   **Mock Exams ({mock_exams}/100)**: " +
                ("Strong quiz baseline. Good performance under mock constraints." if mock_exams >= 70
                 else "Mock baseline is low. Focus on taking simulated prep tests to get comfortable with question formats.") + "\n\n"
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
            f"- Average Sleep Hours: {sleep_hours} hours/day\n"
            f"- Weekly LMS Logins: {lms_logins}\n"
            f"- Latest Mock Exam Score: {mock_exams}/100\n"
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


