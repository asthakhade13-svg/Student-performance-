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
import sqlite3
import threading
from models.lstm_model import train_pytorch_model, StudentTransformerLSTM, get_seq_and_static_data
from models.personalization_manager import apply_personalization, train_personalized_head
from models.rag_vector_store import LocalVectorStore

app = Flask(__name__, static_folder='static')

# Initialize RAG Vector Store
vector_store = LocalVectorStore()

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REGISTRY_PATH = os.path.join(MODELS_DIR, 'registry.json')
CSV_PATH = os.path.join(BASE_DIR, 'student_data.csv')
DB_PATH = os.path.join(MODELS_DIR, 'student_records.db')

# Concurrency thread control locks for active retraining
training_lock = threading.Lock()

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
        
    default_notes = [
        "Needs academic support, low mock exam marks and attendance.",
        "Consistent performer, show steady improvements weekly.",
        "Excellent attention to assignments, sleep patterns are stable.",
        "Outstanding student, active participant in class discussions.",
        "Critical alert: high burnout risk, extremely low sleep hours logged.",
        "Strong assignment completion, maintains high attendance.",
        "Top of the class, demonstrates solid mastery of concepts.",
        "Inconsistent study hours. Missed two assignments in week 3.",
        "Brilliant performance across all metrics, highly motivated.",
        "Average score predictions, attendance needs a slight boost.",
        "Good progress. Demonstrates strong LMS interaction.",
        "Active LMS participation, steady previous marks.",
        "High burnout warning: student exhibits low sleep patterns.",
        "Solid coursework completion, very high final scores expected.",
        "Exhibits great quiz retention, attendance remains stable.",
        "Low mock exams focus, needs assignment review warnings.",
        "Demonstrates solid temporal improvement trends.",
        "Needs study pattern revision, low study hours noted."
    ]
    
    data["final_score"] = final_score
    data["notes"] = default_notes
    return data

def init_sqlite_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_data'")
    table_exists = cursor.fetchone()
    
    upgrade_needed = False
    if table_exists:
        cursor.execute("PRAGMA table_info(student_data)")
        cols = [col[1] for col in cursor.fetchall()]
        if "notes" not in cols:
            upgrade_needed = True
            
    if not table_exists or upgrade_needed:
        df = pd.DataFrame(generate_default_sequence_data())
        df.to_sql("student_data", conn, if_exists="replace", index=False)
        conn.commit()
        for school in ["alpha", "beta", "gamma"]:
            cursor.execute(f"DROP TABLE IF EXISTS student_data_{school}")
        conn.commit()
        
    for school in ["alpha", "beta", "gamma"]:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='student_data_{school}'")
        silo_exists = cursor.fetchone()
        if not silo_exists:
            df = pd.read_sql_query("SELECT * FROM student_data", conn)
            if school == "alpha":
                silo_df = df[df.index % 3 == 0].reset_index(drop=True)
            elif school == "beta":
                silo_df = df[df.index % 3 == 1].reset_index(drop=True)
            else:
                silo_df = df[df.index % 3 == 2].reset_index(drop=True)
            silo_df.to_sql(f"student_data_{school}", conn, if_exists="replace", index=False)
            conn.commit()
            
    conn.close()

def load_or_create_data():
    init_sqlite_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM student_data", conn)
    except Exception:
        df = pd.DataFrame(generate_default_sequence_data())
        df.to_sql("student_data", conn, if_exists="replace", index=False)
    conn.close()
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
    # Auto-replicate single inputs to weekly inputs to support simple frontend forms
    for feat in ["study_hours", "sleep_hours", "lms_logins", "assignments_completed", "mock_exams"]:
        if feat in data:
            for w in range(1, 5):
                if f"{feat}_w{w}" not in data:
                    data[f"{feat}_w{w}"] = data[feat]
                    
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


PERSONALIZATION_FILE = os.path.join("models", "personalization.json")

def load_personalization():
    if not os.path.exists(PERSONALIZATION_FILE):
        return {}
    try:
        with open(PERSONALIZATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_student_bias(student_id):
    biases = load_personalization()
    return float(biases.get(student_id, 0.0))

def save_student_bias(student_id, bias_value):
    biases = load_personalization()
    biases[student_id] = float(bias_value)
    try:
        with open(PERSONALIZATION_FILE, "w", encoding="utf-8") as f:
            json.dump(biases, f, indent=4)
    except Exception as e:
        print("Error saving personalization bias:", e)


def background_train_task():
    global training_lock
    if not training_lock.acquire(blocking=False):
        print("[MLOps] Background training already in progress. Skipping.")
        return
    try:
        print("[MLOps] Background training started...")
        df = load_or_create_data()
        train_model(df)
        print("[MLOps] Background training completed successfully.")
    except Exception as e:
        print("[MLOps] Error in background training:", e)
    finally:
        training_lock.release()

def trigger_background_training():
    t = threading.Thread(target=background_train_task)
    t.daemon = True
    t.start()


def get_model_and_stats():
    df = load_or_create_data()
    registry = load_registry()
    active_ver = registry.get("active_version")
    
    if active_ver:
        model_entry = next((item for item in registry["history"] if item["version"] == active_ver), None)
        if model_entry and os.path.exists(model_entry["path"]):
            try:
                payload = joblib.load(model_entry["path"])
                model = StudentTransformerLSTM()
                model.load_state_dict(payload["model_state"])
                model.eval()
                return model, payload["scaler_x"], payload["scaler_y"], model_entry["mae"], model_entry["r2"], active_ver
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
        
        model, scaler_x, scaler_y, mae, r2, active_ver = get_model_and_stats()
        
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
        
        seq_val, _, _ = get_seq_and_static_data(student_df)
        N_val, T_val, F_val = seq_val.shape
        seq_val_flat = seq_val.reshape(-1, F_val)
        seq_val_scaled = scaler_x.transform(seq_val_flat).reshape(N_val, T_val, F_val)
        
        student_id = data.get('student_id', 'default_student').strip() or 'default_student'
        notes_text = data.get('notes', '').strip()
        
        # Qualitative lexicon sentiment analyzer for predictable model shifts
        POSITIVE_KEYWORDS = ["excellent", "outstanding", "brilliant", "progress", "motivated", "active", "consistent", "good", "strong", "great", "stable", "high"]
        NEGATIVE_KEYWORDS = ["struggle", "dropout", "disengaged", "burnout", "alert", "missed", "low", "poor", "warning", "critical", "deprivation"]
        
        def get_lexicon_sentiment(text):
            if not text:
                return 0.0
            tokens = text.lower().replace('.', ' ').replace(',', ' ').split()
            score = 0.0
            for w in tokens:
                if any(kw in w for kw in POSITIVE_KEYWORDS):
                    score += 1.0
                elif any(kw in w for kw in NEGATIVE_KEYWORDS):
                    score -= 1.0
            return max(-3.0, min(3.0, score))
            
        sentiment_shift = get_lexicon_sentiment(notes_text) * 1.5
        
        from models.lstm_model import prepare_text_tensors
        idx_tensor, off_tensor = prepare_text_tensors([notes_text])
        
        with torch.no_grad():
            pred_reg, pred_clf, attn_weights = model(torch.tensor(seq_val_scaled, dtype=torch.float32), idx_tensor, off_tensor)
            reg_unscaled = scaler_y.inverse_transform(pred_reg.numpy())
            global_predicted_score = round(float(reg_unscaled[0][0]) + sentiment_shift, 2)
            global_predicted_score = max(0.0, min(100.0, global_predicted_score))
            
            # Map burnout risk levels
            probs = torch.softmax(pred_clf, dim=1).numpy()[0]
            burnout_idx = int(np.argmax(probs))
            burnout_labels = ["Low", "Medium", "High"]
            burnout_risk = burnout_labels[burnout_idx]
            
            # Slice attention weights
            attn_list = attn_weights.numpy()[0].flatten().tolist()
            
            # Apply meta-learning student-specific neural layer personalization
            has_personalization = apply_personalization(model, student_id)
            if has_personalization:
                pred_reg_pers, _, _ = model(torch.tensor(seq_val_scaled, dtype=torch.float32), idx_tensor, off_tensor)
                reg_pers_unscaled = scaler_y.inverse_transform(pred_reg_pers.numpy())
                predicted_score = round(float(reg_pers_unscaled[0][0]) + sentiment_shift, 2)
                predicted_score = max(0.0, min(100.0, predicted_score))
            else:
                predicted_score = global_predicted_score
                
            bias = round(predicted_score - global_predicted_score, 2)
            
        # Calculate SHAP explanations
        import shap
        df = load_or_create_data()
        X_all = df[FEATURE_COLS]
        
        def shap_predict_fn(X_np):
            df_temp = pd.DataFrame(X_np, columns=FEATURE_COLS)
            seq, _, _ = get_seq_and_static_data(df_temp)
            N_t, T_t, F_t = seq.shape
            seq_flat_t = seq.reshape(-1, F_t)
            seq_scaled = scaler_x.transform(seq_flat_t).reshape(N_t, T_t, F_t)
            
            notes_list = [notes_text] * N_t
            idx_temp, off_temp = prepare_text_tensors(notes_list)
            
            with torch.no_grad():
                preds_tensor, _, _ = model(torch.tensor(seq_scaled, dtype=torch.float32), idx_temp, off_temp)
                preds_unscaled = scaler_y.inverse_transform(preds_tensor.numpy())
            return (preds_unscaled.flatten() + sentiment_shift)
            
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

        base_value = round(float(explainer.expected_value), 2)

        return jsonify({
            "success": True,
            "predicted_score": predicted_score,
            "global_predicted_score": global_predicted_score,
            "personalization_bias": bias,
            "grade": grade,
            "grade_class": grade_class,
            "mae": mae,
            "r2": r2,
            "active_version": active_ver,
            "explanations": explanations,
            "base_value": base_value,
            "burnout_risk": burnout_risk,
            "attention_weights": attn_list
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
            "previous_marks": float(data['previous_marks']),
            "notes": data.get("notes", "").strip()
        }
        for w in range(1, 5):
            new_row[f"study_hours_w{w}"] = float(data[f"study_hours_w{w}"])
            new_row[f"sleep_hours_w{w}"] = float(data[f"sleep_hours_w{w}"])
            new_row[f"lms_logins_w{w}"] = float(data[f"lms_logins_w{w}"])
            new_row[f"assignments_completed_w{w}"] = float(data[f"assignments_completed_w{w}"])
            new_row[f"mock_exams_w{w}"] = float(data[f"mock_exams_w{w}"])
        new_row["final_score"] = float(data['final_score'])
        
        school_id = data.get("school_id", "alpha").strip().lower()
        if school_id not in ["alpha", "beta", "gamma"]:
            school_id = "alpha"
            
        conn = sqlite3.connect(DB_PATH)
        new_df = pd.DataFrame([new_row])
        # Write to local school silo database
        new_df.to_sql(f"student_data_{school_id}", conn, if_exists="append", index=False)
        # Also write to global student_data
        new_df.to_sql("student_data", conn, if_exists="append", index=False)
        conn.commit()
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM student_data")
        total_count = cursor.fetchone()[0]
        conn.close()
        
        # Trigger asynchronous background training
        trigger_background_training()
        
        return jsonify({
            "success": True, 
            "message": f"Student added to school '{school_id.upper()}' silo. Background model retraining started.", 
            "total": total_count
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
        
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("student_data", conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
        
        trigger_background_training()
        
        return jsonify({
            "success": True, 
            "message": "Student deleted successfully. Background model retraining started.", 
            "total": len(df)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    try:
        model, scaler_x, scaler_y, mae, r2, active_ver = get_model_and_stats()
        
        df = load_or_create_data()
        X_all = df[FEATURE_COLS]
        
        # 1. Scale-Optimized SHAP background (1 sample median prototype) and subset evaluation (at most 10)
        import shap
        median_x = X_all.median(axis=0).values.reshape(1, -1)
        background = pd.DataFrame(median_x, columns=FEATURE_COLS)
        sample_subset = X_all.sample(n=min(10, len(X_all)), random_state=42)
        
        def shap_predict_fn(X_np):
            df_temp = pd.DataFrame(X_np, columns=FEATURE_COLS)
            seq, _, _ = get_seq_and_static_data(df_temp)
            N_t, T_t, F_t = seq.shape
            seq_flat_t = seq.reshape(-1, F_t)
            seq_scaled = scaler_x.transform(seq_flat_t).reshape(N_t, T_t, F_t)
            with torch.no_grad():
                preds_tensor, _, _ = model(torch.tensor(seq_scaled, dtype=torch.float32))
                preds_unscaled = scaler_y.inverse_transform(preds_tensor.numpy())
            return preds_unscaled.flatten()
            
        explainer = shap.KernelExplainer(shap_predict_fn, background)
        shap_vals = explainer.shap_values(sample_subset, l1_reg=False)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
            
        importance = np.mean(np.abs(shap_vals), axis=0).tolist()
        
        # 2. Compute Class Cohort Metrics
        total_students = len(df)
        avg_attendance = round(float(df["attendance"].mean()), 1) if total_students else 0.0
        avg_prev_marks = round(float(df["previous_marks"].mean()), 1) if total_students else 0.0
        
        cohort_scores = []
        high_burnout_count = 0
        
        seq_data, _, _ = get_seq_and_static_data(df)
        N_val, T_val, F_val = seq_data.shape
        seq_flat_val = seq_data.reshape(-1, F_val)
        seq_scaled_val = scaler_x.transform(seq_flat_val).reshape(N_val, T_val, F_val)
        
        with torch.no_grad():
            pred_reg, pred_clf, _ = model(torch.tensor(seq_scaled_val, dtype=torch.float32))
            reg_unscaled = scaler_y.inverse_transform(pred_reg.numpy())
            
            for idx in range(N_val):
                score = round(float(reg_unscaled[idx][0]), 2)
                score = max(0.0, min(100.0, score))
                cohort_scores.append(score)
                
                probs = torch.softmax(pred_clf[idx], dim=0).numpy()
                burnout_idx = int(np.argmax(probs))
                if burnout_idx == 2:  # High burnout risk category
                    high_burnout_count += 1
                    
        avg_predicted_score = round(float(np.mean(cohort_scores)), 1) if cohort_scores else 0.0
        burnout_pct = round((high_burnout_count / total_students) * 100, 1) if total_students else 0.0
        
        # 3. Compute Local School Silo Data Sizes
        school_sizes = {}
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for school in ["alpha", "beta", "gamma"]:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM student_data_{school}")
                school_sizes[school] = cursor.fetchone()[0]
            except Exception:
                school_sizes[school] = 0
        conn.close()
        
        return jsonify({
            "success": True,
            "features": FEATURE_COLS,
            "importance": importance,
            "mae": mae,
            "r2": r2,
            "total_students": total_students,
            "avg_attendance": avg_attendance,
            "avg_prev_marks": avg_prev_marks,
            "avg_predicted_score": avg_predicted_score,
            "burnout_pct": burnout_pct,
            "school_sizes": school_sizes,
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


@app.route('/api/federated-train', methods=['POST'])
def federated_train():
    try:
        from models.federated_learning import run_federated_rounds
        data = request.get_json() or {}
        rounds = int(data.get("rounds", 3))
        epochs = int(data.get("epochs", 5))
        lr = float(data.get("lr", 0.01))
        noise_scale = float(data.get("noise_scale", 0.01))
        
        success, logs, active_ver = run_federated_rounds(rounds=rounds, epochs=epochs, lr=lr, noise_scale=noise_scale)
        if success:
            return jsonify({
                "success": True,
                "message": f"Federated model {active_ver} trained and active!",
                "logs": logs,
                "active_version": active_ver
            })
        else:
            return jsonify({"success": False, "error": "Federated training failed.", "logs": logs}), 400
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
        burnout_risk = data.get('burnout_risk', 'Low')
        explanations = data.get('explanations', [])

        # RAG Pipeline: Extract features with negative SHAP impacts
        negative_exps = [e for e in explanations if e.get('impact', 0) < 0]
        negative_exps = sorted(negative_exps, key=lambda x: x.get('impact', 0))
        top_neg_features = [e.get('feature') for e in negative_exps[:2]]

        # Fallback queries based on low baseline metrics
        if not top_neg_features:
            fallbacks = []
            if attendance < 85: fallbacks.append('attendance')
            if sleep_hours < 7.0: fallbacks.append('sleep_hours')
            if study_hours < 6.0: fallbacks.append('study_hours')
            if assignments_completed < 8: fallbacks.append('assignments_completed')
            if mock_exams < 70: fallbacks.append('mock_exams')
            top_neg_features = fallbacks[:2]

        if not top_neg_features:
            top_neg_features = ['study_hours', 'mock_exams']

        # Map base features to semantic query strings
        query_map = {
            'study_hours': "study hours active study routines daily study Chapter 1 Pages 12-45",
            'sleep_hours': "sleep hours rest wellness cognitive fatigue Chapter 2 Pages 46-88",
            'lms_logins': "LMS logins logins week digital engagement Chapter 3 Pages 89-130",
            'assignments_completed': "assignments completed homework completion Chapter 4 Pages 131-180",
            'mock_exams': "mock exams quiz baseline exam prep test strategy Chapter 5 Pages 181-220",
            'attendance': "attendance lecture class slides syllabus links",
            'previous_marks': "previous marks exam grades study foundations"
        }

        query_texts = []
        for feat in top_neg_features:
            base_feat = feat
            for suffix in ['_w1', '_w2', '_w3', '_w4']:
                if feat.endswith(suffix):
                    base_feat = feat[:-3]
                    break
            q_str = query_map.get(base_feat, feat)
            query_texts.append((feat, q_str))

        # Retrieve documents from LocalVectorStore
        retrieved_docs = []
        for feat, q_str in query_texts:
            matches = vector_store.query(q_str, top_k=1)
            for m in matches:
                source_label = m["metadata"]["source"]
                retrieved_docs.append(f"Struggle Zone Reference ({feat} - from {source_label}):\n{m['content']}")

        if not retrieved_docs:
            matches = vector_store.query("syllabus textbook", top_k=2)
            for m in matches:
                source_label = m["metadata"]["source"]
                retrieved_docs.append(f"General Context (from {source_label}):\n{m['content']}")

        rag_context = "\n\n".join(retrieved_docs)

        # Check if API Key is configured
        if not GEMINI_API_KEY:
            # Format RAG recommendations callout in mock report
            rag_callout = ""
            if retrieved_docs:
                rag_callout = "#### 📖 Recommended Reading & Study Resources (RAG matched):\n"
                for doc in retrieved_docs:
                    rag_callout += f"> * {doc.replace('# ', '').strip().replace(chr(10), ' ')}\n"
                rag_callout += "\n"

            burnout_alert = (
                "CRITICAL WARNING: High risk of student burnout! You must prioritize rest, increase sleep hours, and take frequent study breaks." if burnout_risk == "High"
                else "Caution: Moderate risk of burnout detected. Balance study hours with relaxation and ensure regular sleep." if burnout_risk == "Medium"
                else "Excellent! You are maintaining a healthy study-life balance."
            )
            mock_plan = (
                "### Personalized AI Academic Advisory Report\n\n"
                "> *Notice: Gemini API Key is not set in environment variables. Displaying simulated RAG-augmented study plan.*\n\n"
                "#### 1. Strength & Risk Factor Analysis\n"
                f"{rag_callout}"
                f"*   **Burnout Risk Alert ({burnout_risk})**: {burnout_alert}\n"
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
            f"- AI Predicted Final Exam Score: {predicted_score}/100\n"
            f"- Predicted Student Burnout Risk Category: {burnout_risk}\n\n"
            "--- Struggle Zone Context (Retrieval-Augmented Reference Material) ---\n"
            f"{rag_context}\n\n"
            "Include these exact three sections, using headers:\n"
            "1. Strength & Risk Factor Analysis: Analyze their metrics. Compare parameters and point out major areas causing lower scores vs. areas keeping them afloat. Make sure to address their predicted Burnout Risk category and offer advice accordingly.\n"
            "2. Custom 4-Week Action Planner: A specific, week-by-week plan detailing study subjects or practices to raise their grade. You MUST reference the exact textbook page numbers, chapter names, or lecture slide links retrieved in the Struggle Zone Context above to make this plan highly actionable.\n"
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


@app.route('/api/log-feedback', methods=['POST'])
def log_feedback():
    try:
        data = request.get_json()
        student_id = data.get('student_id', 'default_student').strip() or 'default_student'
        actual_score = float(data.get('actual_score'))
        predicted_score = float(data.get('predicted_score'))
        features = data.get('features', {})
        
        # 1. Online Gradient Descent Meta-Learning fine-tuning
        model, scaler_x, scaler_y, mae, r2, active_ver = get_model_and_stats()
        
        # Build features DataFrame and sequence
        features_dict = {}
        for col in FEATURE_COLS:
            features_dict[col] = [float(features.get(col, 0.0))]
        feat_df = pd.DataFrame(features_dict)
        seq_val, _, _ = get_seq_and_static_data(feat_df)
        N_val, T_val, F_val = seq_val.shape
        seq_val_flat = seq_val.reshape(-1, F_val)
        seq_val_scaled = scaler_x.transform(seq_val_flat).reshape(N_val, T_val, F_val)
        
        # Apply current personalization if exists, then fine-tune online
        apply_personalization(model, student_id)
        train_personalized_head(model, seq_val_scaled, actual_score, scaler_y, student_id)
        
        # Compute new personalization bias offset
        with torch.no_grad():
            pred_reg_pers, _, _ = model(torch.tensor(seq_val_scaled, dtype=torch.float32))
            reg_pers_unscaled = scaler_y.inverse_transform(pred_reg_pers.numpy())
            new_personalized_score = round(float(reg_pers_unscaled[0][0]), 2)
            new_personalized_score = max(0.0, min(100.0, new_personalized_score))
            
            # Load clean global model to calculate un-personalized score
            global_model, _, _, _, _, _ = get_model_and_stats()
            pred_reg_glob, _, _ = global_model(torch.tensor(seq_val_scaled, dtype=torch.float32))
            reg_glob_unscaled = scaler_y.inverse_transform(pred_reg_glob.numpy())
            new_global_score = round(float(reg_glob_unscaled[0][0]), 2)
            
            new_bias = round(new_personalized_score - new_global_score, 2)

        # 2. Append new student record to SQLite for active retraining
        new_row = {}
        for col in FEATURE_COLS:
            new_row[col] = float(features.get(col, 0.0))
        new_row['final_score'] = actual_score
        
        conn = sqlite3.connect(DB_PATH)
        new_df = pd.DataFrame([new_row])
        new_df.to_sql("student_data", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()
        
        # 3. Trigger asynchronous background training
        trigger_background_training()
        
        return jsonify({
            "success": True,
            "new_bias": new_bias,
            "message": f"Feedback logged successfully. Personalized bias updated to {new_bias:+.2f}. Model retraining queued in background."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    load_or_create_data()
    app.run(debug=False, port=5000)


