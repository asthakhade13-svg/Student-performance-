# 🎓 Student Performance Predictor (AI-MLOps Dashboard)

An adaptive, end-to-end Machine Learning platform designed to forecast student final marks, analyze academic risk levels, and generate customized counseling recommendations.

---

## ⚡ Features

*   **Deep Learning Predictor**: LSTM-Transformer sequential network forecasting final grades based on 4-week habits (attendance, study hours, sleep, LMS activity, assignments, and mock scores).
*   **MARL Academic Advisor**: Cooperative Multi-Agent Reinforcement Learning modeling student compliance trajectories and advising customized study actions.
*   **Agentic RAG Counseling**: Semantic lookup utilizing a local vector store to pull pedagogical reference articles for academic reports.
*   **Federated Learning**: Differential Privacy-preserving local silo model aggregation (FedAvg) across simulated school nodes ("alpha", "beta", "gamma").
*   **Autonomous Data Ingest**: Background directory file-watcher scanning `incoming_data/` for CSV/JSON drops, and a direct REST API ingestion webhook.
*   **Glassmorphic Dashboard UI**: Translucent glass-card layout with mouse cursor specular shine, hovering shimmer sweep beams, pastel gradients, and animated pop-up tips.

---

## 🛠️ REST API Endpoints

*   `POST /api/predict`: Predicts final marks and computes week-by-week self-attention attributions.
*   `POST /api/generate-advice`: Returns RAG counseling action plans and runs RL compliance simulations.
*   `POST /api/adapt/ingest-webhook`: Autonomously adds external student records and queues retraining.


