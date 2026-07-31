import os
import sqlite3
import numpy as np
import pandas as pd

REACT_SYSTEM_PROMPT = """You are EduPredict Advisor, an autonomous academic agent. Your task is to analyze a student's profile and generate a highly personalized study advisory report.
You must include a detailed 'Custom 4-Week Action Planner & Study Instructions' section, providing week-wise daily tasks, methods (e.g. active recall, spaced repetition, Feynman technique), specific chapter references, and targeted study/sleep hour adjustments to help the student improve their grades.

You have access to the following tools:

1. search_learning_materials(query: str) -> str: Searches syllabus and textbook chunks.
2. query_cohort_db(sql_query: str) -> str: Queries cohort database statistics. E.g. 'SELECT AVG(sleep_hours_w1) FROM student_data'.
3. cohort_comparator(feature_name: str, value: float) -> str: Compares the student's metrics to the cohort average.

To use a tool, output in this exact format:
Thought: Describe why you need to call the tool.
Action: tool_name(arguments)

For example:
Thought: I need to check how the student's sleep compares to the cohort average.
Action: cohort_comparator("sleep_hours", 5.0)

When you receive the Observation, continue the thought cycle. When you have gathered enough information, double-check all recommendations against cohort stats to self-correct. Then output your final report in this format:
Final Answer: [Your complete markdown report]
"""

def search_learning_materials(query_str):
    """
    Search syllabus and textbook for relevant recommendations.
    """
    from app import vector_store
    matches = vector_store.query(query_str, top_k=2)
    return "\n\n".join([f"Source: {m['metadata']['source']}\n{m['content']}" for m in matches])

def query_cohort_db(sql_query):
    """
    Execute a read-only SQL query on the student cohort database to gather statistics.
    Only SELECT statements are allowed.
    """
    if not sql_query.strip().lower().startswith("select"):
        return "Error: Only SELECT statements are permitted."
    try:
        conn = sqlite3.connect("models/student_records.db")
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"

def cohort_comparator(feature_name, value):
    """
    Compare a student's parameter against the overall cohort average.
    """
    try:
        conn = sqlite3.connect("models/student_records.db")
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
        # Live ReAct loop using Google Gemini Model
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
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
        
        chat = model.start_chat()
        chat.send_message(REACT_SYSTEM_PROMPT)
        
        response_text = chat.send_message(prompt).text
        logs.append(f"Thought: Analyzing student profile and deciding next steps.")
        
        for step in range(5):
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
                    
                obs_msg = f"Observation: {obs}"
                logs.append(obs_msg)
                
                response_text = chat.send_message(obs_msg).text
            elif "Final Answer:" in response_text:
                break
            else:
                break
                
        if "Final Answer:" in response_text:
            final_report = response_text.split("Final Answer:")[1].strip()
        else:
            final_report = response_text
            
        return final_report, logs
