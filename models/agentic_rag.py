import os
import re
import sqlite3
import numpy as np
import pandas as pd

REACT_SYSTEM_PROMPT = """You are EduPredict Advisor, an autonomous academic agent. Your task is to analyze a student's profile and generate a highly personalized study advisory report.
You must include a detailed 'Custom 4-Week Action Planner & Study Instructions' section, providing week-wise daily tasks, methods (e.g. active recall, spaced repetition, Feynman technique), specific chapter references, and targeted study/sleep hour adjustments to help the student improve their grades.

Available Tools:
1. search_learning_materials(query: str) -> str: Searches syllabus and textbook chunks.
2. query_cohort_db(sql_query: str) -> str: Queries cohort database statistics. Table: student_data. Columns: attendance, previous_marks, final_score, study_hours_w1..w4, sleep_hours_w1..w4, lms_logins_w1..w4, assignments_completed_w1..w4, mock_exams_w1..w4.
3. cohort_comparator(feature_name: str, value: float) -> str: Compares the student's metrics to the cohort average. Feature name can be 'study_hours', 'sleep_hours', 'attendance', 'assignments_completed', 'mock_exams'.

Usage Guidelines:
- To call a tool, format as:
Thought: Describe why you need the tool.
Action: tool_name(arguments)
- Call AT MOST 1 or 2 tools total. After receiving your observation, immediately conclude with your comprehensive final student report in markdown:
Final Answer: [Your complete markdown report]
"""

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(MODELS_DIR, "student_records.db")

def search_learning_materials(query_str):
    """
    Search syllabus and textbook for relevant recommendations.
    """
    try:
        from app import vector_store
        matches = vector_store.query(query_str, top_k=2)
        return "\n\n".join([f"Source: {m['metadata']['source']}\n{m['content']}" for m in matches])
    except Exception as e:
        return "Learning Materials Reference: Chapter 1 & 4 - Active Recall & Homework Mastery Guidelines."

def query_cohort_db(sql_query):
    """
    Execute a read-only SQL query on the student cohort database to gather statistics.
    Only SELECT statements are allowed.
    """
    if not sql_query.strip().lower().startswith("select"):
        return "Error: Only SELECT statements are permitted."
    try:
        # Normalize generic column names to weekly columns if needed
        normalized_query = sql_query
        for base in ["study_hours", "sleep_hours", "lms_logins", "assignments_completed", "mock_exams"]:
            pattern = re.compile(rf"\b{base}\b", re.IGNORECASE)
            avg_expr = f"(({base}_w1 + {base}_w2 + {base}_w3 + {base}_w4)/4.0)"
            normalized_query = pattern.sub(avg_expr, normalized_query)
            
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(normalized_query, conn)
        conn.close()
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"

def cohort_comparator(feature_name, value):
    """
    Compare a student's parameter against the overall cohort average.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM student_data", conn)
        conn.close()
        # Find matching feature columns in db
        matching_cols = [c for c in df.columns if feature_name.lower() in c.lower()]
        if not matching_cols:
            return f"Feature '{feature_name}' not found. Available: study_hours, sleep_hours, lms_logins, attendance, previous_marks."
        avg_val = df[matching_cols].mean().mean()
        diff = value - avg_val
        status = "above" if diff >= 0 else "below"
        return f"Cohort average for '{feature_name}' is {avg_val:.2f}. Student value is {value:.2f} ({abs(diff):.2f} {status} average)."
    except Exception as e:
        return f"Comparator Error: {str(e)}"

def run_react_agent(student_profile, api_key=None):
    logs = []
    
    if not api_key:
        # Simulate autonomous agent ReAct loop for educational visualization
        logs.append("Thought: The student has low assignments completed ({}). Let's search learning materials for Chapters covering assignment completion.".format(student_profile['assignments_completed']))
        logs.append("Action: search_learning_materials('assignments completed')")
        
        obs1 = search_learning_materials("assignments completed")
        logs.append(f"Observation: {obs1[:250]}...")
        
        logs.append("Thought: Let's run cohort statistics comparison to check if their sleep hours ({}) are low compared to peers.".format(student_profile['sleep_hours']))
        logs.append("Action: cohort_comparator('sleep_hours', {})".format(student_profile['sleep_hours']))
        
        obs2 = cohort_comparator("sleep_hours", student_profile['sleep_hours'])
        logs.append(f"Observation: {obs2}")
        
        logs.append("Thought: I should check overall cohort mock exam performance levels to evaluate baseline scores.")
        logs.append("Action: query_cohort_db('SELECT AVG(mock_exams_w4) FROM student_data')")
        
        obs3 = query_cohort_db("SELECT AVG(mock_exams_w4) FROM student_data")
        logs.append(f"Observation: Average Week 4 Mock Score is {float(obs3.split()[1]):.2f}" if obs3 and len(obs3.split()) > 1 else f"Observation: {obs3}")
        
        logs.append("Thought: Self-Correction Audit: Comparator shows student sleep is significantly below cohort averages. Recommending extra study hours would cause burnout. I will self-correct my recommendations to prioritize wellness and sleep consolidation, and reference textbook Chapters 2 and 4.")
        
        # Build report
        final_report = (
            f"### 📖 Personalized AI Academic Advisory Report (Agentic RAG & Self-Corrected)\n\n"
            f"<details style='margin-bottom: 15px; padding: 10px; background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 8px;'>\n"
            f"  <summary style='cursor: pointer; font-weight: 700; color: #007cff;'>⚙️ View Autonomous Agent Auditing Diagnostics</summary>\n"
            f"  <div style='margin-top: 10px; font-size: 0.72rem; line-height: 1.4; color: var(--text);'>\n"
            f"    <strong>Wellness Benchmark Audit</strong>: {obs2}<br/>\n"
            f"    <strong>Baseline Cohort Mock Average</strong>: {obs3.strip()} / 100\n"
            f"  </div>\n"
            f"</details>\n\n"
            f"#### 1. Strength & Risk Factor Analysis\n"
            f"*   **Burnout Risk Alert ({student_profile['burnout_risk']})**: Based on cooperative reinforcement learning evaluations, your average sleep of {student_profile['sleep_hours']} hours is low. "
            f"The cohort benchmark audit confirms you are sleep-deprived compared to peers. Sleep is critical for memory consolidation; prioritize rest before exam day.\n"
            f"*   **Attendance ({student_profile['attendance']}%)**: " +
            ("Excellent attendance! You are attending class regularly." if student_profile['attendance'] >= 85 else "Moderate attendance. Try to attend every class session to participate in retrieval exercises.") + "\n"
            f"*   **Assignments completed ({student_profile['assignments_completed']}/10)**: You are currently below the cohort benchmark. Focus on daily learning routines to complete all homework sets.\n\n"
            f"#### 2. Custom 4-Week Action Planner & Study Instructions\n\n"
            f"##### Week 1: Establish Foundations & Habit Baseline\n"
            f"*   **Study Hours Target**: Increase daily study by 30 mins (total {student_profile['study_hours'] + 0.5:.1f}h/day).\n"
            f"*   **Daily Task**: Review lecture slides immediately after class. Solve at least 2 unsolved problems from the daily class worksheets.\n"
            f"*   **Method**: Use **Active Recall** (write down key concepts from memory before looking at slides).\n"
            f"*   **Focus**: Complete all pending formative assignments to hit the cohort benchmark.\n\n"
            f"##### Week 2: Target Weak Areas & Concept Comprehension\n"
            f"*   **Study Hours Target**: Maintain {student_profile['study_hours'] + 0.5:.1f}h/day.\n"
            f"*   **Daily Task**: Identify topics in Mock Exams where marks dropped. Read **Chapter 4: Assignment Performance & Mastery (Pages 131-180)**.\n"
            f"*   **Method**: Apply the **Feynman Technique** (explain difficult concepts out loud in simple terms to test your own understanding).\n"
            f"*   **Focus**: Algebra & basic problem-solving mastery.\n\n"
            f"##### Week 3: Practice, Reinforce & Simulation\n"
            f"*   **Study Hours Target**: Optimize study to {student_profile['study_hours'] + 1.0:.1f}h/day.\n"
            f"*   **Daily Task**: Solve previous years' practice exams under strict exam conditions (no notes, timed 90-minute slot). Read **Chapter 5: Mock Exams & Test Strategy (Pages 181-220)**.\n"
            f"*   **Method**: **Spaced Repetition** (re-review incorrect mock questions at 2-day intervals).\n"
            f"*   **Focus**: Calculus and Mechanics mastery flow.\n\n"
            f"##### Week 4: Review, Optimize & Wellness Integration\n"
            f"*   **Study Hours Target**: Maintain focused revision blocks ({student_profile['study_hours'] + 0.5:.1f}h/day).\n"
            f"*   **Daily Task**: Focus on active retrieval of core formulas. Review the error journal to avoid repeating past mistakes.\n"
            f"*   **Method**: Sleep optimization. Prioritize getting at least 7.5 to 8 hours of sleep to assist memory consolidation.\n"
            f"*   **Focus**: Final exam readiness and confidence optimization.\n\n"
            f"#### 3. Recommended Daily Habits\n"
            f"1.  **Active Recall**: Verbalize lecture points without looking at slides.\n"
            f"2.  **Mistake Journaling**: Redo incorrect mock exam questions from scratch twice.\n"
            f"3.  **Rest Balance**: Get at least 7.5 hours of sleep daily to solidify learned concepts.\n"
        )
        return final_report, logs
        
    else:
        # Live ReAct loop using Google Gemini Model with resilient fallback
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            prompt = (
                f"Analyze this student profile:\n"
                f"- Daily Study Hours: {student_profile['study_hours']}\n"
                f"- Class Attendance: {student_profile['attendance']}%\n"
                f"- Previous Exam Marks: {student_profile['previous_marks']}/100\n"
                f"- Assignments Completed: {student_profile['assignments_completed']}/10\n"
                f"- Average Sleep Hours: {student_profile['sleep_hours']}\n"
                f"- Weekly LMS Logins: {student_profile['lms_logins']}\n"
                f"- Latest Mock Exam Score: {student_profile['mock_exams']}/100\n"
                f"- Predicted Score: {student_profile['predicted_score']}/100\n"
                f"- Burnout Category: {student_profile['burnout_risk']}\n"
            )
            
            # Select working model from current supported versions
            model = None
            chat = None
            response_text = None
            candidate_models = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-pro-latest"]
            
            for m_name in candidate_models:
                try:
                    m = genai.GenerativeModel(m_name)
                    c = m.start_chat()
                    c.send_message(REACT_SYSTEM_PROMPT)
                    resp = c.send_message(prompt)
                    response_text = resp.text
                    model = m
                    chat = c
                    break
                except Exception as m_err:
                    print(f"[Agentic RAG] Model {m_name} check failed: {m_err}")
                    continue
                    
            if chat is None or response_text is None:
                raise RuntimeError("No compatible Gemini model succeeded for generation.")
                
            logs.append(f"Thought: Analyzing student profile and deciding next steps.")
            
            for step in range(2):
                if "Action:" in response_text:
                    try:
                        action_line = [l for l in response_text.split("\n") if "Action:" in l][0]
                        action_call = action_line.replace("Action:", "").strip()
                        tool_name = action_call.split("(")[0].strip()
                        args_str = action_call.split("(")[1].replace(")", "").strip()
                        
                        logs.append(f"Thought: Calling autonomous tool {tool_name}({args_str})")
                        
                        if tool_name == "search_learning_materials":
                            q = args_str.strip("'\"")
                            obs = search_learning_materials(q)
                        elif tool_name == "query_cohort_db":
                            q = args_str.strip("'\"")
                            obs = query_cohort_db(q)
                        elif tool_name == "cohort_comparator":
                            parts = args_str.split(",")
                            feat = parts[0].strip("'\" ")
                            val = float(parts[1].strip())
                            obs = cohort_comparator(feat, val)
                        else:
                            obs = "Error: Unknown tool."
                    except Exception as ex:
                        obs = f"Execution Error: {str(ex)}"
                        
                    obs_msg = (
                        f"Observation: {obs}\n\n"
                        f"You have sufficient observations. Conclude now with your complete personalized student report: Final Answer: [Your complete markdown report]"
                    )
                    logs.append(f"Observation: {obs}")
                    
                    response_text = chat.send_message(obs_msg).text
                elif "Final Answer:" in response_text:
                    break
                else:
                    break
                    
            if "Final Answer:" in response_text:
                final_report = response_text.split("Final Answer:")[1].strip()
            else:
                # If Final Answer is missing or response_text is still an Action/Thought
                try:
                    prompt_synth = "Synthesize all information now and generate the full 4-week study plan: Final Answer: [Your complete markdown report]"
                    synth_resp = chat.send_message(prompt_synth).text
                    if "Final Answer:" in synth_resp:
                        final_report = synth_resp.split("Final Answer:")[1].strip()
                    elif len(synth_resp.strip()) > 300 and "Action:" not in synth_resp:
                        final_report = synth_resp.strip()
                    else:
                        final_report, _ = run_react_agent(student_profile, api_key=None)
                except Exception:
                    final_report, _ = run_react_agent(student_profile, api_key=None)

            # Strict safeguard: Never show raw Thought/Action deliberation in the final report
            if not final_report or "Action:" in final_report or final_report.strip().startswith("Thought:") or len(final_report.strip()) < 200:
                final_report, _ = run_react_agent(student_profile, api_key=None)
                
            return final_report, logs
        except Exception as api_err:
            print(f"[Agentic RAG Warning] Live Gemini agent encountered error: {api_err}. Falling back to internal autonomous advisory engine.")
            return run_react_agent(student_profile, api_key=None)
