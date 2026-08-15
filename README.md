# 🎓 AI-Powered Student Performance Predictor & MLOps Platform

An end-to-end adaptive machine learning platform that forecasts student final marks, quantifies behavioral risk, and generates custom agentic academic counseling plans. The system utilizes sequential deep learning, cooperative reinforcement learning, retrieval-augmented generation (RAG), and privacy-preserving federated training.

---

## 🚀 Key Features & Architecture

### 1. Sequential Deep Learning Predictor (LSTM-Transformer)
*   **Model Core**: Processes 4-week temporal sequences of student habits (attendance, study hours, sleep balance, LMS logins, completed assignments, and mock scores) alongside static covariates.
*   **Attention Mechanisms**: Leverages self-attention modules to highlight which weeks and parameters had the most critical impact on predicted final marks.
*   **Explainable AI**: Employs SHAP and feature attribution mappings to visualize factor influences for teachers and counselors.

### 2. Multi-Agent RL Cooperative Advisor (MARL)
*   **Game-Theoretic Counseling**: Models academic counseling as a cooperative game between two RL agents:
    *   **Advisor Agent**: Recommends optimal study-rest strategies.
    *   **Student Compliance Agent**: Simulates compliance behaviors.
*   **Cooperative Projections**: Simulates compliance transitions across a 4-week timeline, projecting final outcomes based on student compliance adjustments.

### 3. Agentic Retrieval-Augmented Generation (RAG)
*   **Siloed Knowledge Base**: Embeds pedagogical books, course guides, and sleep-study balancing documents in a local vector database.
*   **Context-Aware Advice**: Generates hyper-personalized counseling recommendations by retrieving relevant pages and combining them with the student's current weekly metrics.

### 4. Autonomous Adaptive Ingestion Pipeline
*   **Directory File Watcher**: A background daemon monitors the `incoming_data/` directory. Dropping `.csv` or `.json` logs of student profiles triggers auto-ingestion, SQL updates, and incremental background retraining.
*   **REST Webhook Ingest**: Exposes a POST endpoint `/api/adapt/ingest-webhook` allowing external LMS platforms (Canvas, Moodle) to push student profiles directly.

### 5. Privacy-Preserving Federated Learning
*   **Decentralized Training**: Simulates multi-tenant school silos ("alpha", "beta", "gamma") to train local models independently.
*   **Differential Privacy (DP)**: Injects Gaussian noise into local weight matrices before FedAvg aggregation, protecting student records against model inversion attacks.

### 6. Premium Glassmorphism UI
*   **Dynamic Holographic Reflections**: Features flat glossy reflections and highlight colors tracking cursor coordinates.
*   **Metallic Shimmer Sweeps**: A glossy silver beam sweeps across cards on hover to provide premium visual feedback.
*   **High-Contrast Background**: Floating blurred gradient orbs drift behind cards over a pastel fixed Lavender-to-Sky-Blue background gradient.
*   **Interactive Modal Popups**: Shows tips descriptions in elegant glassmorphic overlay pop-ups on list click, supporting outside clicks and ESC key closures.

---

## 🛠️ API Documentation

### `POST /api/predict`
Calculates final grade predictions and feature attributions.
*   **Request Payload**:
    ```json
    {
      "attendance": 85.0,
      "previous_marks": 75.0,
      "study_hours_w1": 4.5, "sleep_hours_w1": 7.0, "lms_logins_w1": 25, "assignments_completed_w1": 8, "mock_exams_w1": 65,
      "study_hours_w2": 5.0, "sleep_hours_w2": 7.5, "lms_logins_w2": 28, "assignments_completed_w2": 9, "mock_exams_w2": 70,
      "study_hours_w3": 5.5, "sleep_hours_w3": 8.0, "lms_logins_w3": 30, "assignments_completed_w3": 10, "mock_exams_w3": 75,
      "study_hours_w4": 6.0, "sleep_hours_w4": 7.0, "lms_logins_w4": 32, "assignments_completed_w4": 9, "mock_exams_w4": 78
    }
    ```

### `POST /api/generate-advice`
Generates RAG academic advisory action items and simulates MARL projections.
*   **Request Payload**: Include the keys from `/api/predict` along with `"enable_rl": true`.

### `POST /api/adapt/ingest-webhook`
Direct ingestion route for external school information systems.
*   **Request Payload**: Single JSON object or array of student records matching prediction features.

---

## 💻 Getting Started

### 1. Install Dependencies
Ensure you have the required packages installed:
```bash
pip install flask numpy pandas torch scikit-learn optuna google-genai sqlite3
```

### 2. Start the Server
Run the Flask application:
```bash
python app.py
```
The server will initialize models, start the background directory watcher, and serve the dashboard locally at **http://127.0.0.1:5000**.
