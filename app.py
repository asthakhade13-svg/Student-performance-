from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
import json
import shutil
import time
from datetime import datetime
import google.generativeai as genai
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import sqlite3
import threading
from models.lstm_model import train_pytorch_model, StudentTransformerLSTM, get_seq_and_static_data, prepare_text_tensors
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
    
    # District demographic indicator (0 = urban/district A, 1 = rural/district B)
    district = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    # Gender demographic indicator (0 = Male/Non-binary, 1 = Female)
    gender = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
    
    study_base = [2.0, 4.0, 6.0, 8.0, 1.0, 5.0, 7.0, 3.0, 9.0, 4.0, 5.0, 5.0, 5.0, 6.0, 3.0, 7.0, 4.0, 2.0]
    sleep_base = [7.0, 6.5, 8.0, 7.5, 5.5, 8.0, 7.0, 7.5, 8.5, 6.0, 7.5, 7.5, 4.0, 8.5, 7.5, 5.0, 8.0, 7.5]
    lms_base = [15, 25, 40, 45, 10, 30, 50, 20, 55, 28, 30, 30, 12, 45, 80, 15, 35, 50]
    assign_base = [4, 6, 8, 9, 2, 7, 10, 5, 10, 6, 7, 7, 8, 9, 5, 6, 7, 4]
    mock_base = [48.0, 68.0, 79.0, 87.0, 38.0, 74.0, 91.0, 58.0, 96.0, 62.0, 70.0, 70.0, 55.0, 92.0, 85.0, 62.0, 75.0, 68.0]
    
    data = {
        "attendance": attendance,
        "previous_marks": previous_marks,
        "district": district,
        "gender": gender
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
        if "notes" not in cols or "district" not in cols or "gender" not in cols:
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
            
    # Initialize student_connections table (GNN Adjacency)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_connections'")
    if not cursor.fetchone():
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_connections (
                student_id_1 TEXT,
                student_id_2 TEXT,
                weight REAL
            )
        """)
        conn.commit()
        # Populate default cohort study group connections
        for i in range(18):
            for j in [i-1, i+1, i-2, i+2]:
                if 0 <= j < 18 and i != j:
                    cursor.execute("INSERT INTO student_connections VALUES (?, ?, ?)", (f"student_{i}", f"student_{j}", 0.5))
        conn.commit()
        
    # Initialize student_quiz_logs table (DKT interactions)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_quiz_logs'")
    if not cursor.fetchone():
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_quiz_logs (
                student_id TEXT,
                week INTEGER,
                skill_id TEXT,
                is_correct INTEGER
            )
        """)
        conn.commit()
        # Populate with default quiz logs for default cohort
        import random
        random.seed(42)
        skills = ["Algebra", "Calculus", "Mechanics"]
        for i in range(18):
            s_id = f"student_{i}"
            success_prob = 0.5 + 0.02 * i
            for w in range(1, 5):
                for skill in skills:
                    is_correct = 1 if random.random() < success_prob else 0
                    cursor.execute("INSERT INTO student_quiz_logs VALUES (?, ?, ?, ?)", (s_id, w, skill, is_correct))
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
    is_zs = data.get('zero_shot')
    for feat in ["study_hours", "sleep_hours", "lms_logins", "assignments_completed", "mock_exams"]:
        for w in range(1, 5):
            if f"{feat}_w{w}" not in data:
                if is_zs:
                    data[f"{feat}_w{w}"] = 0.0
                elif feat in data:
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
    
    mae_mean, mae_std, r2_mean, fairness_dist, fairness_gend = train_pytorch_model(df, model_filepath)
    
    entry = {
        "version": version,
        "path": model_filepath,
        "r2": round(float(r2_mean), 2),
        "mae": round(float(mae_mean), 2),
        "mae_std": round(float(mae_std), 2),
        "data_size": len(df),
        "fairness_district": fairness_dist,
        "fairness_gender": fairness_gend,
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


# DKT Global Model Cache
from models.dkt_model import StudentDKT, train_dkt_model, get_student_mastery
global_dkt_model = StudentDKT()
dkt_path = os.path.join(MODELS_DIR, "dkt_model.pth")

def initialize_dkt_agent():
    global global_dkt_model
    try:
        df_all = load_or_create_data()
        if os.path.exists(dkt_path):
            global_dkt_model.load_state_dict(torch.load(dkt_path))
            global_dkt_model.eval()
            print("[DKT Initialization] Loaded active DKT model checkpoint.")
        else:
            train_dkt_model(global_dkt_model, DB_PATH, epochs=30)
            torch.save(global_dkt_model.state_dict(), dkt_path)
            global_dkt_model.eval()
            print("[DKT Initialization] Trained and saved baseline DKT model.")
    except Exception as e:
        print(f"[DKT Initialization] Failed to load/train DKT model: {str(e)}")


def background_train_task():
    global training_lock
    if not training_lock.acquire(blocking=False):
        print("[MLOps] Background training already in progress. Skipping.")
        return
    try:
        print("[MLOps] Background training started...")
        df = load_or_create_data()
        train_model(df)
        
        # Retrain DKT
        try:
            train_dkt_model(global_dkt_model, DB_PATH, epochs=30)
            torch.save(global_dkt_model.state_dict(), dkt_path)
            print("[MLOps] Background DKT model retraining completed.")
        except Exception as e:
            print("[MLOps] Failed to retrain DKT in background:", e)
            
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


global_rl_agent = None

def initialize_rl_agent():
    global global_rl_agent
    try:
        from models.rl_advisor import train_rl_advisor
        df_all = load_or_create_data()
        
        # Load model and stats once to optimize prediction speed inside the RL training loop
        model, scaler_x, scaler_y, _, _, _ = get_model_and_stats()
        model.eval()
        
        def predict_state(state):
            # Layout matching get_seq_and_static_data:
            # study_hours, sleep_hours, lms_logins, assignments_completed, mock_exams, attendance, previous_marks
            row_data = np.array([state[2], state[3], state[4], state[5], state[6], state[0], state[1]])
            seq_data = np.tile(row_data, (4, 1))
            seq_scaled = scaler_x.transform(seq_data).reshape(1, 4, 7)
            
            with torch.no_grad():
                pred, _, _ = model(torch.tensor(seq_scaled, dtype=torch.float32))
                score = scaler_y.inverse_transform(pred.numpy())[0][0]
            return float(score)

        global_rl_agent = train_rl_advisor(df_all, predict_state, epochs=120)
    except Exception as e:
        print(f"[RL Initialization] Failed to train DQN agent: {str(e)}")


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


def predict_with_uncertainty(model, seq_scaled, idx_tensor, off_tensor, scaler_y, sentiment_shift, num_samples=50):
    model.train() # Enable MC Dropout (Bayesian Neural Network approach)
    preds = []
    for _ in range(num_samples):
        with torch.no_grad():
            pred_reg, _, _ = model(torch.tensor(seq_scaled, dtype=torch.float32), idx_tensor, off_tensor)
            reg_unscaled = scaler_y.inverse_transform(pred_reg.numpy())
            score = float(reg_unscaled[0][0]) + sentiment_shift
            preds.append(max(0.0, min(100.0, score)))
    model.eval() # Restore evaluation mode
    mean_val = float(np.mean(preds))
    std_val = float(np.std(preds)) * 6.0 # Scale to map dropout variance to realistic marks bounds
    variance_val = float(np.var(preds)) * 36.0 # \sigma^2 = \frac{1}{B} \sum_{b=1}^B (\hat{y}_b - \bar{y})^2
    return round(mean_val, 2), max(0.1, round(std_val, 2)), round(variance_val, 4)


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
        
        is_zs = data.get("zero_shot", False)
        df_all = load_or_create_data()
        
        student_data_dict = {
            "attendance": [attendance],
            "previous_marks": [previous_marks]
        }
        for w in range(1, 5):
            for feat in ["study_hours", "sleep_hours", "lms_logins", "assignments_completed", "mock_exams"]:
                key = f"{feat}_w{w}"
                if is_zs:
                    cohort_avg = float(df_all[key].mean()) if len(df_all) > 0 else 5.0
                    student_data_dict[key] = [cohort_avg]
                else:
                    student_data_dict[key] = [float(data[key])]
            
        student_df = pd.DataFrame(student_data_dict)
        student_df = student_df[FEATURE_COLS]
        
        student_id = data.get('student_id', 'default_student').strip() or 'default_student'
        notes_text = data.get('notes', '').strip()
        
        # 1. Fetch cohort DataFrame
        df_all = load_or_create_data()
        if "student_id" not in df_all.columns:
            df_all["student_id"] = [f"student_{i}" for i in range(len(df_all))]
            
        # Append candidate student row
        cand_row = student_data_dict.copy()
        cand_row["student_id"] = [student_id]
        cand_row["notes"] = [notes_text]
        cand_row["district"] = [int(data.get("district", 0))]
        cand_row["gender"] = [int(data.get("gender", 0))]
        cand_df = pd.DataFrame(cand_row)
        
        df_aug = pd.concat([df_all, cand_df], ignore_index=True)
        
        # Build peer connections in SQLite if they don't exist yet for this student
        try:
            conn_db = sqlite3.connect(DB_PATH)
            c_db = conn_db.cursor()
            c_db.execute("SELECT COUNT(*) FROM student_connections WHERE student_id_1 = ? OR student_id_2 = ?", (student_id, student_id))
            if c_db.fetchone()[0] == 0:
                c_db.execute("SELECT rowid, previous_marks FROM student_data")
                all_rows = c_db.fetchall()
                all_rows = sorted(all_rows, key=lambda r: abs(r[1] - previous_marks))
                connections_added = 0
                for r_idx, r_pm in all_rows:
                    peer_id = f"student_{r_idx - 1}"
                    if peer_id != student_id and connections_added < 2:
                        c_db.execute("INSERT INTO student_connections VALUES (?, ?, ?)", (student_id, peer_id, 0.5))
                        connections_added += 1
                conn_db.commit()
            conn_db.close()
        except Exception as e:
            print("[Predict GNN Connection] Failed to update SQLite student connections:", e)
            
        # Get sequence data for entire augmented cohort
        seq_all, _, _ = get_seq_and_static_data(df_aug)
        N_aug, T_aug, F_aug = seq_all.shape
        seq_all_flat = seq_all.reshape(-1, F_aug)
        seq_all_scaled = scaler_x.transform(seq_all_flat).reshape(N_aug, T_aug, F_aug)
        
        # Load adjacency matrix for augmented cohort
        from models.lstm_model import load_adjacency_matrix
        adj_aug = load_adjacency_matrix(df_aug)
        
        # Build text embedding offsets for all nodes
        notes_all = df_aug["notes"].values
        idx_all, off_all = prepare_text_tensors(notes_all)
        
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
        
        with torch.no_grad():
            pred_reg_all, pred_clf_all, attn_weights_all = model(
                torch.tensor(seq_all_scaled, dtype=torch.float32), 
                idx_all, 
                off_all, 
                adj_aug
            )
            
            # Slice output for candidate student (last row in batch)
            cand_idx = N_aug - 1
            pred_reg = pred_reg_all[cand_idx].unsqueeze(0)
            pred_clf = pred_clf_all[cand_idx].unsqueeze(0)
            attn_weights = attn_weights_all[cand_idx].unsqueeze(0)
            
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
                # Predict with personalized head (message-passed through neighbor graph)
                pred_reg_pers_all, _, _ = model(
                    torch.tensor(seq_all_scaled, dtype=torch.float32), 
                    idx_all, 
                    off_all, 
                    adj_aug
                )
                pred_reg_pers = pred_reg_pers_all[cand_idx].unsqueeze(0)
                reg_pers_unscaled = scaler_y.inverse_transform(pred_reg_pers.numpy())
                predicted_score = round(float(reg_pers_unscaled[0][0]) + sentiment_shift, 2)
                predicted_score = max(0.0, min(100.0, predicted_score))
            else:
                predicted_score = global_predicted_score
                
            bias = round(predicted_score - global_predicted_score, 2)
            
        # Compute Monte Carlo Dropout prediction uncertainty (50 passes)
        idx_cand, off_cand = prepare_text_tensors([notes_text])
        _, uncertainty, variance = predict_with_uncertainty(model, seq_all_scaled[[cand_idx]], idx_cand, off_cand, scaler_y, sentiment_shift, num_samples=50)
            
        # Calculate SHAP explanations
        import shap
        df = load_or_create_data()
        X_all = df[FEATURE_COLS]
        
        def shap_predict_fn(X_np):
            N_t = X_np.shape[0]
            batch_size = 100
            all_preds = []
            
            for idx_b in range(0, N_t, batch_size):
                X_batch = X_np[idx_b : idx_b + batch_size]
                N_batch = X_batch.shape[0]
                
                df_temp = pd.concat([student_df] * N_batch, ignore_index=True)
                for idx_c, col in enumerate(FEATURE_COLS):
                    df_temp[col] = X_batch[:, idx_c]
                
                seq, _, _ = get_seq_and_static_data(df_temp)
                N_b, T_b, F_b = seq.shape
                seq_flat_b = seq.reshape(-1, F_b)
                seq_scaled = scaler_x.transform(seq_flat_b).reshape(N_b, T_b, F_b)
                
                notes_list = [notes_text] * N_b
                idx_temp, off_temp = prepare_text_tensors(notes_list)
                
                with torch.no_grad():
                    adj_batch = torch.eye(N_b, device=seq_scaled.device if hasattr(seq_scaled, 'device') else None)
                    preds_tensor, _, _ = model(
                        torch.tensor(seq_scaled, dtype=torch.float32), 
                        idx_temp, 
                        off_temp, 
                        adj_batch
                    )
                    preds_unscaled = scaler_y.inverse_transform(preds_tensor.numpy())
                all_preds.append(preds_unscaled.flatten() + sentiment_shift)
                
            return np.concatenate(all_preds)
            
        # Use median background profile and limit perturbation samples to make SHAP prediction instant (< 200ms!)
        background_summary = pd.DataFrame([X_all.median()], columns=FEATURE_COLS)
        explainer = shap.KernelExplainer(shap_predict_fn, background_summary)
        shap_vals = explainer.shap_values(student_df, nsamples=80, l1_reg=False)
        
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

        from models.personalization_manager import load_personalized_heads
        heads = load_personalized_heads()
        if student_id in heads:
            profile_status = "One-Shot (Adapted)"
        elif is_zs:
            profile_status = "Zero-Shot (MAML Baseline)"
        else:
            profile_status = "Standard Cohort Baseline"

        return jsonify({
            "success": True,
            "predicted_score": predicted_score,
            "global_predicted_score": global_predicted_score,
            "uncertainty": uncertainty,
            "variance": variance,
            "personalization_bias": bias,
            "grade": grade,
            "grade_class": grade_class,
            "mae": mae,
            "r2": r2,
            "active_version": active_ver,
            "explanations": explanations,
            "base_value": base_value,
            "burnout_risk": burnout_risk,
            "attention_weights": attn_list,
            "profile_status": profile_status
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


@app.route('/api/active-learning-queue', methods=['GET'])
def active_learning_queue():
    try:
        model, scaler_x, scaler_y, mae, r2, active_ver = get_model_and_stats()
        df = load_or_create_data()
        if len(df) == 0:
            return jsonify({"success": True, "queue": []})
            
        queue_records = []
        from models.lstm_model import get_seq_and_static_data, prepare_text_tensors
        from models.personalization_manager import apply_personalization
        import copy
        from models.lstm_model import StudentTransformerLSTM
        
        for idx, row in df.iterrows():
            student_id = row.get("student_id", f"student_{idx}")
            if not student_id or not isinstance(student_id, str):
                student_id = f"student_{idx}"
                
            student_df = pd.DataFrame([row])
            seq, _, _ = get_seq_and_static_data(student_df)
            N, T, F = seq.shape
            seq_flat = seq.reshape(-1, F)
            seq_scaled = scaler_x.transform(seq_flat).reshape(N, T, F)
            
            notes_text = row.get("notes", "")
            if not isinstance(notes_text, str):
                notes_text = ""
                
            idx_tensor, off_tensor = prepare_text_tensors([notes_text])
            
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
            
            local_model = StudentTransformerLSTM()
            local_model.load_state_dict(model.state_dict())
            apply_personalization(local_model, student_id)
            
            mean_score, std_dev, variance = predict_with_uncertainty(local_model, seq_scaled, idx_tensor, off_tensor, scaler_y, sentiment_shift, num_samples=50)
            
            queue_records.append({
                "student_id": student_id,
                "predicted_score": mean_score,
                "uncertainty": std_dev,
                "variance": variance,
                "attendance": float(row.get("attendance", 80.0)),
                "previous_marks": float(row.get("previous_marks", 70.0)),
                "notes": notes_text,
                "priority": "High" if std_dev >= 2.0 else ("Medium" if std_dev >= 1.0 else "Low")
            })
            
        queue_records = sorted(queue_records, key=lambda x: x["uncertainty"], reverse=True)
        return jsonify({"success": True, "queue": queue_records})
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


def ingest_dataframe(incoming_df):
    required_cols = [
        "attendance", "previous_marks",
        "study_hours_w1", "sleep_hours_w1", "lms_logins_w1", "assignments_completed_w1", "mock_exams_w1",
        "study_hours_w2", "sleep_hours_w2", "lms_logins_w2", "assignments_completed_w2", "mock_exams_w2",
        "study_hours_w3", "sleep_hours_w3", "lms_logins_w3", "assignments_completed_w3", "mock_exams_w3",
        "study_hours_w4", "sleep_hours_w4", "lms_logins_w4", "assignments_completed_w4", "mock_exams_w4",
        "final_score"
    ]
    
    # Verify columns exist
    missing_cols = [col for col in required_cols if col not in incoming_df.columns]
    if missing_cols:
        print(f"[Adaptive Ingestion] Missing columns in dataframe: {missing_cols}")
        return False
        
    try:
        # Cast datatype
        for col in required_cols:
            incoming_df[col] = pd.to_numeric(incoming_df[col], errors='coerce')
        incoming_df = incoming_df.dropna(subset=required_cols)
        
        if len(incoming_df) == 0:
            print("[Adaptive Ingestion] Empty valid student records after numeric coercion.")
            return False
            
        # Optional metadata fields like notes and school_id
        if "notes" not in incoming_df.columns:
            incoming_df["notes"] = "Autonomously Ingested Profile"
        if "school_id" not in incoming_df.columns:
            incoming_df["school_id"] = "alpha"
            
        conn = sqlite3.connect(DB_PATH)
        for _, row in incoming_df.iterrows():
            school_id = str(row.get("school_id", "alpha")).strip().lower()
            if school_id not in ["alpha", "beta", "gamma"]:
                school_id = "alpha"
                
            student_row = {col: float(row[col]) for col in required_cols}
            student_row["notes"] = str(row.get("notes", "Autonomously Ingested Profile")).strip()
            
            # Insert into database
            row_df = pd.DataFrame([student_row])
            row_df.to_sql(f"student_data_{school_id}", conn, if_exists="append", index=False)
            row_df.to_sql("student_data", conn, if_exists="append", index=False)
            
            # Also append to student_data.csv to synchronize!
            try:
                csv_df = pd.read_csv(CSV_PATH)
                csv_df = pd.concat([csv_df, row_df], ignore_index=True)
                csv_df.to_csv(CSV_PATH, index=False)
            except Exception as csv_err:
                print(f"[Adaptive Ingestion] CSV backup sync failed: {str(csv_err)}")
            
        conn.commit()
        conn.close()
        
        # Trigger model retraining
        trigger_background_training()
        return True
    except Exception as e:
        print(f"[Adaptive Ingestion] Database write failure: {str(e)}")
        return False

def start_adaptive_file_watcher():
    incoming_dir = os.path.join(BASE_DIR, "incoming_data")
    archive_dir = os.path.join(BASE_DIR, "ingested_archive")
    
    if not os.path.exists(incoming_dir):
        os.makedirs(incoming_dir)
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        
    def watch_loop():
        print(f"[Adaptive Ingestion] Directory watcher listening on '{incoming_dir}'...")
        while True:
            try:
                files = [f for f in os.listdir(incoming_dir) if f.endswith('.csv') or f.endswith('.json')]
                for file in files:
                    filepath = os.path.join(incoming_dir, file)
                    print(f"[Adaptive Ingestion] New file detected: {file}")
                    
                    success = False
                    # Parse CSV or JSON
                    if file.endswith('.csv'):
                        try:
                            incoming_df = pd.read_csv(filepath)
                            success = ingest_dataframe(incoming_df)
                        except Exception as parse_err:
                            print(f"[Adaptive Ingestion] CSV parse failed for {file}: {str(parse_err)}")
                    elif file.endswith('.json'):
                        try:
                            with open(filepath, 'r') as json_f:
                                json_data = json.load(json_f)
                            if isinstance(json_data, dict):
                                json_data = [json_data]
                            incoming_df = pd.DataFrame(json_data)
                            success = ingest_dataframe(incoming_df)
                        except Exception as parse_err:
                            print(f"[Adaptive Ingestion] JSON parse failed for {file}: {str(parse_err)}")
                            
                    # Archive the parsed file
                    if success:
                        # Move to archive directory (handling name collisions)
                        base_name, ext = os.path.splitext(file)
                        dest_path = os.path.join(archive_dir, f"{base_name}_{int(time.time())}{ext}")
                        shutil.move(filepath, dest_path)
                        print(f"[Adaptive Ingestion] Successfully ingested and archived to {dest_path}")
                    else:
                        # Move to failed file name format to prevent endless loops
                        failed_path = os.path.join(incoming_dir, f"failed_{int(time.time())}_{file}")
                        if os.path.exists(filepath):
                            os.rename(filepath, failed_path)
                            print(f"[Adaptive Ingestion] Ingestion failed. Renamed to {failed_path}")
            except Exception as loop_err:
                print(f"[Adaptive Ingestion] Loop error: {str(loop_err)}")
                
            time.sleep(5)
            
    t = threading.Thread(target=watch_loop, daemon=True)
    t.start()

@app.route('/api/adapt/ingest-webhook', methods=['POST'])
def adapt_ingest_webhook():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"success": False, "error": "Empty payload"}), 400
            
        if isinstance(payload, dict):
            payload = [payload]
            
        incoming_df = pd.DataFrame(payload)
        success = ingest_dataframe(incoming_df)
        
        if success:
            return jsonify({
                "success": True, 
                "message": "Student records successfully ingested autonomously. Incremental model retraining triggered."
            })
        else:
            return jsonify({"success": False, "error": "Validation error or empty dataset after type casting."}), 400
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


@app.route('/api/counterfactual-recourse', methods=['POST'])
def counterfactual_recourse():
    try:
        data = request.get_json() or {}
        target_score = float(data.get('target_score', 85.0))
        target_score = max(50.0, min(100.0, target_score))
        
        model, scaler_x, scaler_y, mae, r2, active_ver = get_model_and_stats()
        from models.counterfactual_engine import compute_counterfactual_recourse
        
        recourse_result = compute_counterfactual_recourse(
            model=model,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            current_features=data,
            target_score=target_score
        )
        return jsonify({"success": True, "recourse": recourse_result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/contrastive-train', methods=['POST'])
def contrastive_train():
    try:
        from models.contrastive_learning import train_contrastive_pretraining
        df_all = load_or_create_data()
        model, scaler_x, scaler_y, mae, r2, active_ver = get_model_and_stats()
        
        data = request.get_json() or {}
        epochs = int(data.get("epochs", 25))
        
        success, logs = train_contrastive_pretraining(df_all, model, scaler_x, epochs=epochs)
        if success:
            return jsonify({
                "success": True,
                "message": "Self-Supervised Contrastive pre-training completed successfully!",
                "logs": logs
            })
        else:
            return jsonify({"success": False, "error": "Contrastive training failed.", "logs": logs}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/dkt/mastery/<student_id>', methods=['GET'])
def dkt_mastery(student_id):
    try:
        from models.dkt_model import get_student_mastery
        mastery = get_student_mastery(global_dkt_model, student_id, DB_PATH)
        return jsonify({
            "success": True,
            "student_id": student_id,
            "mastery": mastery
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/dkt/log-quiz', methods=['POST'])
def dkt_log_quiz():
    try:
        data = request.get_json() or {}
        student_id = data.get('student_id', 'default_student').strip() or 'default_student'
        skill_id = data.get('skill_id', 'Algebra').strip()
        is_correct = int(data.get('is_correct', 1))
        week = int(data.get('week', 4))
        
        if skill_id not in ["Algebra", "Calculus", "Mechanics"]:
            return jsonify({"success": False, "error": "Invalid skill ID. Choose Algebra, Calculus, or Mechanics."}), 400
            
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO student_quiz_logs VALUES (?, ?, ?, ?)", (student_id, week, skill_id, is_correct))
        conn.commit()
        conn.close()
        
        # Incremental online fine-tuning step for DKT model
        from models.dkt_model import load_student_interactions
        skills_seq, correct_seq = load_student_interactions(student_id, DB_PATH)
        
        if len(skills_seq) >= 2:
            global_dkt_model.train()
            optimizer = torch.optim.Adam(global_dkt_model.parameters(), lr=0.005)
            criterion = nn.BCELoss()
            
            skill_tensor = torch.tensor([skills_seq[:-1]], dtype=torch.long)
            correct_tensor = torch.tensor([correct_seq[:-1]], dtype=torch.float32)
            
            target_skills = skills_seq[1:]
            target_correctness = correct_seq[1:]
            
            for epoch in range(5):
                optimizer.zero_grad()
                probs, _ = global_dkt_model(skill_tensor, correct_tensor)
                
                seq_len = len(target_skills)
                preds = torch.zeros(seq_len)
                for t in range(seq_len):
                    preds[t] = probs[0, t, target_skills[t]]
                    
                targets = torch.tensor(target_correctness, dtype=torch.float32)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                
            global_dkt_model.eval()
            torch.save(global_dkt_model.state_dict(), dkt_path)
            
        from models.dkt_model import get_student_mastery
        mastery = get_student_mastery(global_dkt_model, student_id, DB_PATH)
        
        return jsonify({
            "success": True,
            "message": f"Quiz interaction recorded for {skill_id}.",
            "mastery": mastery
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
        burnout_risk = data.get('burnout_risk', 'Low')
        explanations = data.get('explanations', [])
        
        # Calculate Cooperative MARL Sequential Recommendations
        rl_markdown = ""
        enable_rl = data.get('enable_rl', True)
        try:
            if enable_rl and global_rl_agent is not None:
                from models.rl_advisor import ACTIONS, STUDENT_BEHAVIORS, transition_marl_state
                curr_state = np.array([
                    attendance, previous_marks, study_hours, sleep_hours, lms_logins, assignments_completed, mock_exams
                ])
                
                model, scaler_x, scaler_y, _, _, _ = get_model_and_stats()
                model.eval()
                
                def local_predict(st):
                    row_data = np.array([st[2], st[3], st[4], st[5], st[6], st[0], st[1]])
                    seq_data = np.tile(row_data, (4, 1))
                    seq_scaled = scaler_x.transform(seq_data).reshape(1, 4, 7)
                    with torch.no_grad():
                        pred, _, _ = model(torch.tensor(seq_scaled, dtype=torch.float32))
                        score = scaler_y.inverse_transform(pred.numpy())[0][0]
                    return float(score)

                score_t = local_predict(curr_state)
                rl_timeline = []
                for week in range(1, 5):
                    # Advisor suggests action
                    a_adv = global_rl_agent.select_advisor_action(curr_state, epsilon=0.0)
                    act_adv = ACTIONS[a_adv]
                    
                    # Student decides action based on recommendation
                    a_stud = global_rl_agent.select_student_action(curr_state, a_adv, epsilon=0.0)
                    act_stud = STUDENT_BEHAVIORS[a_stud]
                    
                    next_state = transition_marl_state(curr_state, a_adv, a_stud)
                    score_next = local_predict(next_state)
                    reward = score_next - score_t
                    
                    rl_timeline.append({
                        "week": week,
                        "adv_action_name": act_adv["name"],
                        "stud_action_name": act_stud["name"],
                        "details": f"Study: {next_state[2]:.1f}h/day, Sleep: {next_state[3]:.1f}h/day, LMS Logins: {int(next_state[4])}/wk",
                        "predicted_score": round(score_next, 2),
                        "improvement": round(reward, 2)
                    })
                    curr_state = next_state
                    score_t = score_next
                
                if rl_timeline:
                    rl_markdown = (
                        "### 🎯 Multi-Agent RL Cooperative Intervention & Habit Projections\n"
                        "The counseling advice is modeled as a cooperative game between two RL agents: the **Advisor Agent** (recommending study interventions) and the **Student Agent** (modeling habit compliance behavior):\n\n"
                    )
                    for step in rl_timeline:
                        sign = "+" if step["improvement"] >= 0 else ""
                        rl_markdown += (
                            f"*   **Week {step['week']} Simulation**:\n"
                            f"    *   *Advisor Recommendation*: **{step['adv_action_name']}**\n"
                            f"    *   *Student Compliance Focus*: **{step['stud_action_name']}**\n"
                            f"    *   *Cooperative State*: {step['details']}\n"
                            f"    *   *Projected Grade*: **{step['predicted_score']} / 100** ({sign}{step['improvement']:.2f} marks change)\n"
                        )
                    rl_markdown += "\n"
        except Exception as rl_err:
            print(f"[RL Advisor API] Trajectory simulation failed: {str(rl_err)}")

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

        # Agentic RAG ReAct Loop Execution
        from models.agentic_rag import run_react_agent
        student_profile = {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "assignments_completed": assignments_completed,
            "sleep_hours": sleep_hours,
            "lms_logins": lms_logins,
            "mock_exams": mock_exams,
            "predicted_score": predicted_score,
            "burnout_risk": burnout_risk
        }
        
        final_advice, agent_logs = run_react_agent(student_profile, api_key=GEMINI_API_KEY)
        
        if rl_markdown:
            final_advice = final_advice + "\n---\n\n" + rl_markdown
            
        return jsonify({
            "success": True,
            "advice": final_advice,
            "agent_logs": agent_logs,
            "is_mock": not bool(GEMINI_API_KEY)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/log-feedback', methods=['POST'])
def log_feedback():
    try:
        data = request.get_json()
        student_id = data.get('student_id', 'default_student').strip() or 'default_student'
        actual_score = float(data.get('actual_score'))
        
        # Fallbacks for missing predicted_score and features (e.g. when logging from Active Learning queue or zero-shot scripts)
        pred_raw = data.get('predicted_score')
        predicted_score = float(pred_raw) if pred_raw is not None else 75.0
        
        features = data.get('features')
        if not features:
            df_all = load_or_create_data()
            match_row = pd.DataFrame()
            if 'student_id' in df_all.columns:
                match_row = df_all[df_all['student_id'] == student_id]
            else:
                if student_id.startswith('student_'):
                    try:
                        idx = int(student_id.split('_')[1])
                        if 0 <= idx < len(df_all):
                            match_row = df_all.iloc[[idx]]
                    except Exception:
                        pass
            if len(match_row) > 0:
                features = match_row.iloc[0].to_dict()
            else:
                features = {}
                for col in FEATURE_COLS:
                    features[col] = float(df_all[col].mean()) if len(df_all) > 0 else 5.0
        
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
        new_row['district'] = int(data.get('district', features.get('district', 0)))
        new_row['gender'] = int(data.get('gender', features.get('gender', 0)))
        
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


def initialize_app():
    try:
        load_or_create_data()
        initialize_dkt_agent()
        initialize_rl_agent()
        start_adaptive_file_watcher()
    except Exception as e:
        print(f"[Startup Warning] Initialization error: {e}")

# Run initialization when module is imported (supports Gunicorn & direct python execution)
initialize_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


