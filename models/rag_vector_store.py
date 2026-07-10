import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalVectorStore:
    def __init__(self, knowledge_dir="knowledge_base"):
        self.knowledge_dir = knowledge_dir
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.documents = []
        self.metadata = []
        self.tfidf_matrix = None
        
        # Ensure directory exists and populate defaults if empty
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            self._create_default_docs()
        elif len(os.listdir(self.knowledge_dir)) == 0:
            self._create_default_docs()
            
        self.load_documents()

    def _create_default_docs(self):
        syllabus_content = """# Lecture Syllabus & Slides Catalog
*   **Week 1 Lecture: Foundations of Learning & Study Habits**
    *   Syllabus Topics: Time management, retrieval practice, active recall.
    *   Slide Reference Link: [Lecture 1 Slides](https://course-portal.edu/slides/week1.pdf)
    *   Core Homework: Homework 1 - Time Log Analysis.
*   **Week 2 Lecture: Deep Concept Mapping & Memory Retention**
    *   Syllabus Topics: Spaced repetition, memory consolidation curves, sleep impact.
    *   Slide Reference Link: [Lecture 2 Slides](https://course-portal.edu/slides/week2.pdf)
    *   Core Homework: Homework 2 - Memory Map Design.
*   **Week 3 Lecture: Quantitative Problem Solving & Practical Coding**
    *   Syllabus Topics: Application exercises, assignments, error debugging logs.
    *   Slide Reference Link: [Lecture 3 Slides](https://course-portal.edu/slides/week3.pdf)
    *   Core Homework: Homework 3 - Mock Exam Practice Set.
*   **Week 4 Lecture: Exam Review & Preparation Strategy**
    *   Syllabus Topics: Final exam review, stress management, retrieval heuristics.
    *   Slide Reference Link: [Lecture 4 Slides](https://course-portal.edu/slides/week4.pdf)
    *   Core Homework: Homework 4 - Mock Exam Review.
"""
        with open(os.path.join(self.knowledge_dir, "syllabus.md"), "w", encoding="utf-8") as f:
            f.write(syllabus_content.strip())

        textbook_content = """# Course Textbook: Academic Performance & Science of Learning
*   **Chapter 1: Active Study Routines (Pages 12-45)**
    *   Textbook Topics: Study velocity, daily habits, optimizing study hours.
    *   Focus: High-impact study vs. passive reading.
*   **Chapter 2: The Role of Rest and Cognitive Fatigue (Pages 46-88)**
    *   Textbook Topics: Sleep cycles, memory retention, cognitive burnout prevention.
    *   Focus: Getting 7-8 hours sleep daily.
*   **Chapter 3: Interactive Learning Platforms & LMS (Pages 89-130)**
    *   Textbook Topics: Digital engagement, LMS logins, lecture note reviews.
    *   Focus: Consistent daily digital logins.
*   **Chapter 4: Assignment Performance & Mastery (Pages 131-180)**
    *   Textbook Topics: Formative assignments, quiz baseline, exam simulation.
    *   Focus: Completing all homework sets.
*   **Chapter 5: Mock Exams & Test Strategy (Pages 181-220)**
    *   Textbook Topics: Mock exams, time constraints, testing anxiety.
    *   Focus: Exam simulations and sample question drills.
"""
        with open(os.path.join(self.knowledge_dir, "textbook.md"), "w", encoding="utf-8") as f:
            f.write(textbook_content.strip())

    def load_documents(self):
        self.documents = []
        self.metadata = []
        
        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith(".md") or filename.endswith(".txt"):
                filepath = os.path.join(self.knowledge_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
                    continue
                    
                # Split content into smaller sections based on list catalog items
                sections = content.split("\n*   ")
                header = sections[0]
                for section in sections[1:]:
                    chunk = header + "\n*   " + section
                    self.documents.append(chunk)
                    self.metadata.append({"source": filename})
                    
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def query(self, query_str, top_k=2):
        if not self.documents or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query_str])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.02:  # Safe low relevance threshold for small dataset
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(scores[idx])
                })
        return results
