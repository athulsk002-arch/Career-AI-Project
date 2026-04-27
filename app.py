from flask import Flask, render_template, request, redirect, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import time
import pdfplumber
import requests
import json
import functools

# ─────────────────────────────────────────────
# SKILLS DATABASE
# ─────────────────────────────────────────────

skills_db = {
    "software developer": [
        "python", "java", "c++", "data structures", "algorithms", "sql", "git"
    ],
    "full stack developer": [
        "html", "css", "javascript", "react", "node", "python", "sql", "git", "flask"
    ],
    "web developer": [
        "html", "css", "javascript", "react", "node", "flask", "sql"
    ],
    "data analyst": [
        "python", "sql", "excel", "pandas", "data visualization", "statistics"
    ],
    "cyber security": [
        "networking", "linux", "python", "sql injection", "penetration testing", "cryptography"
    ],
    "ui/ux designer": [
        "figma", "ui", "ux", "wireframing", "css", "user research", "prototyping"
    ]
}

ROLES = list(skills_db.keys())


# ─────────────────────────────────────────────
# SKILL DETECTION & CAREER RECOMMENDATION
# ─────────────────────────────────────────────

def detect_skills(resume_text):
    resume_text = resume_text.lower()
    all_skills  = {skill for job in skills_db for skill in skills_db[job]}
    return [skill for skill in all_skills if skill in resume_text]


def recommend_career(detected_skills):
    scores = {
        job: sum(1 for skill in required if skill in detected_skills)
        for job, required in skills_db.items()
    }
    return max(scores, key=scores.get)


def missing_skills(job, detected_skills):
    return [s for s in skills_db.get(job, []) if s not in detected_skills]


# ─────────────────────────────────────────────
# CORE AI HELPER
# ─────────────────────────────────────────────

def call_claude(prompt, max_tokens=1000):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json"
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "messages":   [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        return data["content"][0]["text"].strip()
    except Exception as e:
        print(f"Claude API error: {e}")
        return None


# ─────────────────────────────────────────────
# AI: Generate MCQ for a level
# ─────────────────────────────────────────────

def ai_generate_mcq(role, level, count):
    """
    level: "basic" | "intermediate" | "critical"
    Returns list of { question, options: {A,B,C,D}, correct }
    """
    level_desc = {
        "basic":        f"fundamental, beginner-friendly questions about {role} concepts",
        "intermediate": f"applied, practical questions requiring working knowledge of {role}",
        "critical":     f"scenario-based, analytical questions requiring deep {role} expertise"
    }

    prompt = f"""Generate exactly {count} multiple choice questions for a {role} assessment.
Level: {level_desc.get(level, level)}

Rules:
- Each question must have exactly 4 options: A, B, C, D
- One option must be clearly correct
- Questions must be specific to {role} only
- No repeated questions
- Keep questions concise (1-2 sentences max)

Respond ONLY with a valid JSON array, no extra text:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct": "A"
  }}
]"""

    result = call_claude(prompt, max_tokens=2000)
    if result:
        try:
            clean     = result.replace("```json", "").replace("```", "").strip()
            questions = json.loads(clean)
            if isinstance(questions, list) and len(questions) > 0:
                return questions[:count]
        except Exception as e:
            print(f"MCQ parse error ({level}): {e}")

    return get_fallback_mcq(role, level, count)


def ai_generate_descriptive(role, count=3):
    """
    Generate practical programming/descriptive tasks.
    Returns list of { question }
    """
    prompt = f"""Generate exactly {count} practical programming or descriptive tasks for a {role} candidate.

Rules:
- Tasks should test if the candidate can write real code or describe real solutions
- Each task should be answerable in writing (no IDE needed)
- Test: functions, syntax knowledge, and practical problem solving
- Be specific to {role} skills
- Keep each task description to 2-3 sentences

Respond ONLY with a valid JSON array:
[
  {{
    "question": "Task description here"
  }}
]"""

    result = call_claude(prompt, max_tokens=800)
    if result:
        try:
            clean = result.replace("```json", "").replace("```", "").strip()
            tasks = json.loads(clean)
            if isinstance(tasks, list) and len(tasks) > 0:
                return tasks[:count]
        except Exception as e:
            print(f"Descriptive parse error: {e}")

    return get_fallback_descriptive(role)


# ─────────────────────────────────────────────
# AI: Revision notes
# ─────────────────────────────────────────────

def ai_revision_notes(role, level, correct, total):
    pct    = round((correct / total) * 100) if total > 0 else 0
    missed = total - correct

    prompt = f"""A {role} candidate just failed the {level} assessment.
They got {correct} out of {total} correct ({pct}%), missing {missed} questions.

Write concise revision notes to help them improve. Include:
1. The key topics they should review for {role} at this level
2. 2-3 specific concepts they likely struggled with
3. A quick summary/cheat sheet of the most important points
4. Encouragement to retry

Keep it practical, clear, and under 300 words. Use simple formatting with line breaks."""

    result = call_claude(prompt, max_tokens=500)
    if result:
        return result

    return f"""📚 Revision Notes for {level}

You scored {pct}% — you need 60% to pass. Here's what to review:

Key topics for {role}:
• Review the fundamental concepts covered in this level
• Practice with real examples and hands-on exercises
• Focus on the areas where you felt uncertain

Tips:
• Re-read core documentation or textbooks for {role}
• Try solving practice problems before retrying
• Take notes on concepts you find difficult

You're making progress! Review these topics and try again."""


# ─────────────────────────────────────────────
# AI: Score descriptive answer
# ─────────────────────────────────────────────

def ai_score_descriptive(role, question, answer):
    if not answer or len(answer.strip()) < 10:
        return 2, "Answer was too short or empty."

    prompt = f"""Evaluate this {role} candidate's answer to a practical task.

Task: {question}
Answer: {answer}

Respond ONLY with JSON:
{{
  "score": <integer 1-10>,
  "feedback": "<2 sentences: one strength, one improvement>"
}}"""

    result = call_claude(prompt, max_tokens=200)
    if result:
        try:
            clean    = result.replace("```json", "").replace("```", "").strip()
            data     = json.loads(clean)
            score    = max(1, min(10, int(data.get("score", 5))))
            feedback = data.get("feedback", "Good effort.")
            return score, feedback
        except Exception:
            pass

    length = len(answer.strip())
    if length > 200: return 7, "Good answer with detail. Include more specific examples next time."
    if length > 80:  return 5, "Decent answer. Add more depth and specifics."
    return 3, "Too short. Explain your solution in more detail."


# ─────────────────────────────────────────────
# AI: Score interview / simulation answer
# ─────────────────────────────────────────────

def ai_score_answer(role, question, answer):
    if not answer or len(answer.strip()) < 5:
        return 2, "Answer was too short."

    prompt = f"""Evaluate this {role} candidate answer.
Question: {question}
Answer: {answer}
Respond ONLY with JSON: {{"score": <1-10>, "feedback": "<two sentences>"}}"""

    result = call_claude(prompt, max_tokens=200)
    if result:
        try:
            clean    = result.replace("```json", "").replace("```", "").strip()
            data     = json.loads(clean)
            score    = max(1, min(10, int(data.get("score", 5))))
            feedback = data.get("feedback", "Good effort.")
            return score, feedback
        except Exception:
            pass

    length = len(answer.strip())
    if length > 150: return 7, "Good answer. Add specific examples next time."
    if length > 60:  return 5, "Decent response. Add more technical depth."
    return 3, "Too short. Explain your reasoning."


# ─────────────────────────────────────────────
# FALLBACK MCQ
# ─────────────────────────────────────────────

def get_fallback_mcq(role, level, count):
    bank = {
        "software developer": {
            "basic": [
                {"question": "What is a variable in programming?", "options": {"A": "A fixed value", "B": "A named storage location", "C": "A function", "D": "A loop"}, "correct": "B"},
                {"question": "Which of these is a loop in Python?", "options": {"A": "switch", "B": "foreach", "C": "for", "D": "repeat"}, "correct": "C"},
                {"question": "What does HTML stand for?", "options": {"A": "HyperText Markup Language", "B": "High Text Machine Language", "C": "HyperTransfer Markup Logic", "D": "Home Tool Markup Language"}, "correct": "A"},
                {"question": "What is a function?", "options": {"A": "A variable type", "B": "A reusable block of code", "C": "A database", "D": "A loop"}, "correct": "B"},
                {"question": "What does Git do?", "options": {"A": "Runs programs", "B": "Manages databases", "C": "Tracks code changes", "D": "Designs interfaces"}, "correct": "C"},
                {"question": "What is an array?", "options": {"A": "A single value", "B": "A collection of values", "C": "A condition", "D": "A file"}, "correct": "B"},
                {"question": "What is a bug in software?", "options": {"A": "A feature", "B": "An error in code", "C": "A test", "D": "A comment"}, "correct": "B"},
                {"question": "What does SQL stand for?", "options": {"A": "Structured Query Language", "B": "Simple Query Logic", "C": "System Query Layer", "D": "Standard Query List"}, "correct": "A"},
                {"question": "What is an IDE?", "options": {"A": "Internet Data Exchange", "B": "Integrated Development Environment", "C": "Internal Design Editor", "D": "Interface Design Engine"}, "correct": "B"},
                {"question": "What is a comment in code?", "options": {"A": "Executable code", "B": "A variable", "C": "Non-executed explanation text", "D": "A function call"}, "correct": "C"},
            ],
            "intermediate": [
                {"question": "What is the time complexity of binary search?", "options": {"A": "O(n)", "B": "O(log n)", "C": "O(n²)", "D": "O(1)"}, "correct": "B"},
                {"question": "What does OOP stand for?", "options": {"A": "Object Oriented Programming", "B": "Open Oriented Process", "C": "Ordered Object Processing", "D": "Optional Object Protocol"}, "correct": "A"},
                {"question": "What is recursion?", "options": {"A": "A loop type", "B": "A function calling itself", "C": "A sorting method", "D": "A data type"}, "correct": "B"},
                {"question": "What is a stack data structure?", "options": {"A": "FIFO order", "B": "LIFO order", "C": "Random order", "D": "Sorted order"}, "correct": "B"},
                {"question": "What is a primary key?", "options": {"A": "The first column", "B": "A unique row identifier", "C": "A foreign reference", "D": "An auto-increment"}, "correct": "B"},
                {"question": "What is inheritance in OOP?", "options": {"A": "Copying code", "B": "A class acquiring properties of another", "C": "A function type", "D": "A database method"}, "correct": "B"},
                {"question": "What is the purpose of an index in a database?", "options": {"A": "To store data", "B": "To speed up queries", "C": "To delete records", "D": "To encrypt data"}, "correct": "B"},
                {"question": "What is a REST API?", "options": {"A": "A database", "B": "A programming language", "C": "An architectural style for web services", "D": "A testing framework"}, "correct": "C"},
                {"question": "What does DRY stand for in programming?", "options": {"A": "Do Repeat Yourself", "B": "Don't Repeat Yourself", "C": "Data Repository Yield", "D": "Dynamic Runtime Yield"}, "correct": "B"},
                {"question": "What is a queue data structure?", "options": {"A": "LIFO order", "B": "FIFO order", "C": "Random order", "D": "Priority order"}, "correct": "B"},
            ],
            "critical": [
                {"question": "Which sorting algorithm has O(n log n) average time complexity?", "options": {"A": "Bubble sort", "B": "Insertion sort", "C": "Merge sort", "D": "Selection sort"}, "correct": "C"},
                {"question": "What is a deadlock in concurrent programming?", "options": {"A": "A slow program", "B": "Two processes blocking each other indefinitely", "C": "A memory leak", "D": "A race condition"}, "correct": "B"},
                {"question": "What is the CAP theorem?", "options": {"A": "A sorting rule", "B": "Consistency, Availability, Partition tolerance tradeoff", "C": "A security protocol", "D": "A UI design principle"}, "correct": "B"},
                {"question": "What is memoization?", "options": {"A": "Writing comments", "B": "Caching results of expensive function calls", "C": "A sorting method", "D": "Memory management"}, "correct": "B"},
                {"question": "In SOLID principles, what does the 'S' stand for?", "options": {"A": "Scalability", "B": "Single Responsibility", "C": "Static Binding", "D": "Sequencing"}, "correct": "B"},
            ],
        },
        "data analyst": {
            "basic": [
                {"question": "What does SQL SELECT do?", "options": {"A": "Delete records", "B": "Retrieve data", "C": "Create tables", "D": "Update records"}, "correct": "B"},
                {"question": "What is a spreadsheet?", "options": {"A": "A database", "B": "A grid-based data tool", "C": "A programming language", "D": "A chart type"}, "correct": "B"},
                {"question": "What does CSV stand for?", "options": {"A": "Comma Separated Values", "B": "Central Server Values", "C": "Code Stored Values", "D": "Column Structured View"}, "correct": "A"},
                {"question": "What is the mean?", "options": {"A": "The middle value", "B": "The most common value", "C": "The average", "D": "The range"}, "correct": "C"},
                {"question": "What is a bar chart used for?", "options": {"A": "Showing trends over time", "B": "Comparing categories", "C": "Showing proportions", "D": "Plotting correlations"}, "correct": "B"},
                {"question": "What is data cleaning?", "options": {"A": "Deleting data", "B": "Fixing errors in data", "C": "Sorting data", "D": "Encrypting data"}, "correct": "B"},
                {"question": "What is a database?", "options": {"A": "A spreadsheet", "B": "Organised collection of data", "C": "A programming language", "D": "A chart"}, "correct": "B"},
                {"question": "What does ETL stand for?", "options": {"A": "Extract Transform Load", "B": "Evaluate Test Logic", "C": "Export Table Layer", "D": "Encode Transfer Link"}, "correct": "A"},
                {"question": "Which Python library is used for data analysis?", "options": {"A": "Flask", "B": "Django", "C": "Pandas", "D": "React"}, "correct": "C"},
                {"question": "What is a null value?", "options": {"A": "Zero", "B": "Missing or unknown data", "C": "A negative number", "D": "An empty string"}, "correct": "B"},
            ],
            "intermediate": [
                {"question": "What is a JOIN in SQL?", "options": {"A": "Splitting a table", "B": "Combining rows from multiple tables", "C": "Deleting duplicates", "D": "Sorting records"}, "correct": "B"},
                {"question": "What is the median?", "options": {"A": "The average", "B": "The middle value when sorted", "C": "The most common value", "D": "The largest value"}, "correct": "B"},
                {"question": "What is data normalisation?", "options": {"A": "Encrypting data", "B": "Organising data to reduce redundancy", "C": "Visualising data", "D": "Deleting duplicates"}, "correct": "B"},
                {"question": "What is standard deviation?", "options": {"A": "The average", "B": "Measure of data spread", "C": "The median", "D": "The maximum value"}, "correct": "B"},
                {"question": "What is a pivot table?", "options": {"A": "A chart type", "B": "A tool to summarise and reorganise data", "C": "A SQL command", "D": "A data type"}, "correct": "B"},
                {"question": "What does GROUP BY do in SQL?", "options": {"A": "Sorts data", "B": "Groups rows sharing a value", "C": "Filters data", "D": "Joins tables"}, "correct": "B"},
                {"question": "What is a scatter plot used for?", "options": {"A": "Comparing categories", "B": "Showing trends", "C": "Showing correlations", "D": "Showing proportions"}, "correct": "C"},
                {"question": "What is an outlier?", "options": {"A": "The average value", "B": "A value far from others", "C": "A missing value", "D": "The median"}, "correct": "B"},
                {"question": "What does VLOOKUP do in Excel?", "options": {"A": "Creates charts", "B": "Looks up a value in a column", "C": "Sorts data", "D": "Filters rows"}, "correct": "B"},
                {"question": "What is data aggregation?", "options": {"A": "Cleaning data", "B": "Summarising multiple values into one", "C": "Visualising data", "D": "Encrypting data"}, "correct": "B"},
            ],
            "critical": [
                {"question": "What is correlation vs causation?", "options": {"A": "They are the same", "B": "Correlation shows relationship; causation shows cause and effect", "C": "Causation is a type of correlation", "D": "They are unrelated concepts"}, "correct": "B"},
                {"question": "What is a Type I error in statistics?", "options": {"A": "False negative", "B": "False positive", "C": "Correct rejection", "D": "Correct acceptance"}, "correct": "B"},
                {"question": "What is A/B testing?", "options": {"A": "Testing two databases", "B": "Comparing two versions to find which performs better", "C": "Testing code errors", "D": "Comparing two datasets"}, "correct": "B"},
                {"question": "What is a p-value?", "options": {"A": "The sample size", "B": "Probability results occurred by chance", "C": "The confidence interval", "D": "The correlation coefficient"}, "correct": "B"},
                {"question": "What is the curse of dimensionality?", "options": {"A": "Too many rows", "B": "Problems arising from high-dimensional data", "C": "Slow database queries", "D": "Missing values"}, "correct": "B"},
            ],
        },
        "web developer": {
            "basic": [
                {"question": "What does HTML stand for?", "options": {"A": "HyperText Markup Language", "B": "High Text Machine Language", "C": "Home Tool Markup Language", "D": "HyperTransfer Markup Logic"}, "correct": "A"},
                {"question": "What does CSS stand for?", "options": {"A": "Computer Style Sheets", "B": "Cascading Style Sheets", "C": "Creative Style Syntax", "D": "Coded Style System"}, "correct": "B"},
                {"question": "Which tag creates a hyperlink in HTML?", "options": {"A": "<link>", "B": "<href>", "C": "<a>", "D": "<url>"}, "correct": "C"},
                {"question": "What is JavaScript used for?", "options": {"A": "Styling pages", "B": "Adding interactivity", "C": "Database queries", "D": "Server management"}, "correct": "B"},
                {"question": "What does DOM stand for?", "options": {"A": "Document Object Model", "B": "Data Object Management", "C": "Dynamic Object Method", "D": "Document Orientation Map"}, "correct": "A"},
                {"question": "What is a responsive website?", "options": {"A": "A fast website", "B": "A website that adapts to screen size", "C": "A secure website", "D": "An animated website"}, "correct": "B"},
                {"question": "Which HTTP method retrieves data?", "options": {"A": "POST", "B": "DELETE", "C": "GET", "D": "PUT"}, "correct": "C"},
                {"question": "What is a browser?", "options": {"A": "A web server", "B": "Software to access the web", "C": "A database", "D": "A programming language"}, "correct": "B"},
                {"question": "What is a URL?", "options": {"A": "A programming language", "B": "A web address", "C": "A file type", "D": "A database"}, "correct": "B"},
                {"question": "What does the <div> tag do in HTML?", "options": {"A": "Creates a link", "B": "Adds an image", "C": "Defines a section", "D": "Creates a table"}, "correct": "C"},
            ],
            "intermediate": [
                {"question": "What is Flexbox in CSS?", "options": {"A": "A JavaScript library", "B": "A layout model for arranging elements", "C": "A font system", "D": "An animation tool"}, "correct": "B"},
                {"question": "What is the box model in CSS?", "options": {"A": "A 3D design tool", "B": "Content, padding, border, margin", "C": "A JavaScript pattern", "D": "A grid system"}, "correct": "B"},
                {"question": "What does async/await do in JavaScript?", "options": {"A": "Creates loops", "B": "Handles asynchronous operations", "C": "Styles elements", "D": "Queries databases"}, "correct": "B"},
                {"question": "What is a CSS media query?", "options": {"A": "A database query", "B": "Applies styles based on screen size", "C": "A JavaScript method", "D": "An HTML attribute"}, "correct": "B"},
                {"question": "What is localStorage?", "options": {"A": "A server database", "B": "Browser-based key-value storage", "C": "A CSS property", "D": "A HTML tag"}, "correct": "B"},
                {"question": "What is a Promise in JavaScript?", "options": {"A": "A variable type", "B": "An object representing a future value", "C": "A CSS animation", "D": "An HTML element"}, "correct": "B"},
                {"question": "What does npm stand for?", "options": {"A": "Node Package Manager", "B": "New Program Module", "C": "Network Protocol Manager", "D": "Node Program Method"}, "correct": "A"},
                {"question": "What is React?", "options": {"A": "A CSS framework", "B": "A JavaScript UI library", "C": "A database", "D": "A server language"}, "correct": "B"},
                {"question": "What is HTTPS?", "options": {"A": "A programming language", "B": "A secure web protocol", "C": "A CSS property", "D": "A JavaScript method"}, "correct": "B"},
                {"question": "What is a CDN?", "options": {"A": "Code Delivery Network", "B": "Content Delivery Network", "C": "Central Data Node", "D": "Core Design Network"}, "correct": "B"},
            ],
            "critical": [
                {"question": "What is the Critical Rendering Path?", "options": {"A": "A JavaScript pattern", "B": "Steps the browser takes to render a page", "C": "A CSS animation sequence", "D": "A server process"}, "correct": "B"},
                {"question": "What is CORS?", "options": {"A": "Cross-Origin Resource Sharing", "B": "Client Object Request System", "C": "Core Object Routing Service", "D": "Common Origin Resource Standard"}, "correct": "A"},
                {"question": "What is lazy loading?", "options": {"A": "Slow code execution", "B": "Loading resources only when needed", "C": "A caching strategy", "D": "A CSS technique"}, "correct": "B"},
                {"question": "What is the difference between == and === in JavaScript?", "options": {"A": "No difference", "B": "=== checks value and type; == only checks value", "C": "== is more strict", "D": "=== is used for strings only"}, "correct": "B"},
                {"question": "What is Web Accessibility (a11y)?", "options": {"A": "Fast loading", "B": "Making web content usable by people with disabilities", "C": "Mobile responsiveness", "D": "SEO optimisation"}, "correct": "B"},
            ],
        },
        "cyber security": {
            "basic": [
                {"question": "What is a firewall?", "options": {"A": "A physical server", "B": "A system that monitors and controls network traffic", "C": "A type of malware", "D": "An encryption method"}, "correct": "B"},
                {"question": "What is phishing?", "options": {"A": "A network protocol", "B": "Tricking users into revealing sensitive info", "C": "A type of encryption", "D": "A virus"}, "correct": "B"},
                {"question": "What does VPN stand for?", "options": {"A": "Virtual Private Network", "B": "Verified Protocol Node", "C": "Variable Port Number", "D": "Virtual Public Network"}, "correct": "A"},
                {"question": "What is malware?", "options": {"A": "A security tool", "B": "Malicious software", "C": "A firewall type", "D": "A programming language"}, "correct": "B"},
                {"question": "What is two-factor authentication?", "options": {"A": "Two passwords", "B": "Verification using two different methods", "C": "Two firewalls", "D": "Two usernames"}, "correct": "B"},
                {"question": "What does HTTPS provide?", "options": {"A": "Faster loading", "B": "Encrypted communication", "C": "Better SEO", "D": "Larger bandwidth"}, "correct": "B"},
                {"question": "What is a DDoS attack?", "options": {"A": "A phishing method", "B": "Overwhelming a server with traffic", "C": "Password cracking", "D": "Data theft"}, "correct": "B"},
                {"question": "What is a password hash?", "options": {"A": "A plain text password", "B": "An encrypted password representation", "C": "A password hint", "D": "A backup password"}, "correct": "B"},
                {"question": "What is social engineering?", "options": {"A": "Building social networks", "B": "Manipulating people to reveal information", "C": "A programming technique", "D": "A network protocol"}, "correct": "B"},
                {"question": "What does CIA stand for in security?", "options": {"A": "Central Intelligence Agency", "B": "Confidentiality, Integrity, Availability", "C": "Code, Integrity, Access", "D": "Control, Inspect, Audit"}, "correct": "B"},
            ],
            "intermediate": [
                {"question": "What is SQL Injection?", "options": {"A": "A database tool", "B": "Inserting malicious SQL into a query", "C": "A firewall technique", "D": "An encryption method"}, "correct": "B"},
                {"question": "What is XSS?", "options": {"A": "Cross-Site Scripting", "B": "Extended Security System", "C": "External Server Script", "D": "Cross-System Security"}, "correct": "A"},
                {"question": "What is a brute force attack?", "options": {"A": "Physical damage", "B": "Trying all possible password combinations", "C": "A social engineering attack", "D": "A DDoS method"}, "correct": "B"},
                {"question": "What is penetration testing?", "options": {"A": "Network monitoring", "B": "Authorised simulated attack to find vulnerabilities", "C": "Password management", "D": "Firewall configuration"}, "correct": "B"},
                {"question": "What is a zero-day vulnerability?", "options": {"A": "A patched vulnerability", "B": "An unknown vulnerability with no fix yet", "C": "A daily security check", "D": "An old vulnerability"}, "correct": "B"},
                {"question": "What is encryption?", "options": {"A": "Deleting data", "B": "Converting data to unreadable format", "C": "Backing up data", "D": "Compressing data"}, "correct": "B"},
                {"question": "What is a man-in-the-middle attack?", "options": {"A": "A physical intrusion", "B": "Intercepting communication between two parties", "C": "A password attack", "D": "A DDoS attack"}, "correct": "B"},
                {"question": "What does OWASP stand for?", "options": {"A": "Open Web Application Security Project", "B": "Official Web Access Security Protocol", "C": "Online Web Application Standard Panel", "D": "Open Wireless Access Security Program"}, "correct": "A"},
                {"question": "What is a digital certificate?", "options": {"A": "A degree", "B": "A digital document verifying identity", "C": "A firewall rule", "D": "A password"}, "correct": "B"},
                {"question": "What is port scanning?", "options": {"A": "Printing documents", "B": "Identifying open ports on a network", "C": "Scanning for malware", "D": "Monitoring traffic"}, "correct": "B"},
            ],
            "critical": [
                {"question": "What is the principle of least privilege?", "options": {"A": "Giving all users admin access", "B": "Users get only the minimum access needed", "C": "Restricting all access", "D": "Sharing access equally"}, "correct": "B"},
                {"question": "What is a rainbow table attack?", "options": {"A": "A colourful UI attack", "B": "Using precomputed hashes to crack passwords", "C": "A network flood", "D": "A social engineering method"}, "correct": "B"},
                {"question": "What is CSRF?", "options": {"A": "Cross-Site Request Forgery", "B": "Central Security Response Framework", "C": "Code Security Risk Factor", "D": "Client-Side Request Filter"}, "correct": "A"},
                {"question": "What is threat modelling?", "options": {"A": "Drawing network diagrams", "B": "Identifying and prioritising potential security threats", "C": "Monitoring logs", "D": "Writing security policies"}, "correct": "B"},
                {"question": "What is defence in depth?", "options": {"A": "A single strong firewall", "B": "Multiple layered security controls", "C": "Deep packet inspection", "D": "Underground server rooms"}, "correct": "B"},
            ],
        },
        "full stack developer": {
            "basic": [
                {"question": "What is a frontend in web development?", "options": {"A": "The database", "B": "The user-facing part of an application", "C": "The server", "D": "The API"}, "correct": "B"},
                {"question": "What is a backend?", "options": {"A": "The UI", "B": "The server-side logic and database", "C": "The CSS", "D": "The browser"}, "correct": "B"},
                {"question": "What does API stand for?", "options": {"A": "Application Programming Interface", "B": "Automated Program Interaction", "C": "Applied Protocol Integration", "D": "Application Process Interface"}, "correct": "A"},
                {"question": "What is a database?", "options": {"A": "A programming language", "B": "Organised collection of structured data", "C": "A web server", "D": "A CSS framework"}, "correct": "B"},
                {"question": "What is Git?", "options": {"A": "A programming language", "B": "A version control system", "C": "A database", "D": "A framework"}, "correct": "B"},
                {"question": "What is Node.js?", "options": {"A": "A CSS framework", "B": "A JavaScript runtime for servers", "C": "A database", "D": "A browser"}, "correct": "B"},
                {"question": "What does HTTP stand for?", "options": {"A": "HyperText Transfer Protocol", "B": "High Transfer Text Protocol", "C": "HyperTransfer Text Process", "D": "Home Transfer Text Protocol"}, "correct": "A"},
                {"question": "What is React?", "options": {"A": "A server language", "B": "A JavaScript UI library", "C": "A database", "D": "A CSS framework"}, "correct": "B"},
                {"question": "What is a package manager?", "options": {"A": "A shipping tool", "B": "A tool to manage code dependencies", "C": "A database tool", "D": "A deployment tool"}, "correct": "B"},
                {"question": "What is JSON?", "options": {"A": "A programming language", "B": "A lightweight data format", "C": "A database type", "D": "A CSS property"}, "correct": "B"},
            ],
            "intermediate": [
                {"question": "What is a REST API?", "options": {"A": "A database", "B": "An architectural style for web services", "C": "A programming language", "D": "A testing tool"}, "correct": "B"},
                {"question": "What is JWT used for?", "options": {"A": "Styling pages", "B": "Authentication tokens", "C": "Database queries", "D": "File compression"}, "correct": "B"},
                {"question": "What is CORS?", "options": {"A": "Cross-Origin Resource Sharing", "B": "Client Object Request System", "C": "Common Origin Routing Service", "D": "Cross-Object Request Standard"}, "correct": "A"},
                {"question": "What is middleware in web development?", "options": {"A": "A UI component", "B": "Software that connects different systems", "C": "A database layer", "D": "A testing framework"}, "correct": "B"},
                {"question": "What is the MVC pattern?", "options": {"A": "Model View Controller", "B": "Multiple View Component", "C": "Module Version Control", "D": "Main View Controller"}, "correct": "A"},
                {"question": "What is a NoSQL database?", "options": {"A": "A database without any queries", "B": "A non-relational database", "C": "A slow database", "D": "A read-only database"}, "correct": "B"},
                {"question": "What is server-side rendering?", "options": {"A": "Rendering on the user's browser", "B": "Generating HTML on the server", "C": "A CSS technique", "D": "A database process"}, "correct": "B"},
                {"question": "What is a webhook?", "options": {"A": "A fishing hook", "B": "HTTP callback triggered by an event", "C": "A UI component", "D": "A database trigger"}, "correct": "B"},
                {"question": "What is containerisation?", "options": {"A": "Packaging apps with their dependencies", "B": "Storing data in containers", "C": "A UI design pattern", "D": "A testing method"}, "correct": "A"},
                {"question": "What is CI/CD?", "options": {"A": "Continuous Integration/Continuous Deployment", "B": "Code Inspection/Code Delivery", "C": "Central Integration/Core Deployment", "D": "Continuous Inspection/Code Deployment"}, "correct": "A"},
            ],
            "critical": [
                {"question": "What is the difference between authentication and authorisation?", "options": {"A": "They are the same", "B": "Authentication verifies identity; authorisation grants permissions", "C": "Authorisation comes first", "D": "Authentication grants permissions"}, "correct": "B"},
                {"question": "What is database sharding?", "options": {"A": "Encrypting a database", "B": "Splitting a database across multiple machines", "C": "Backing up a database", "D": "Indexing a database"}, "correct": "B"},
                {"question": "What is the event loop in Node.js?", "options": {"A": "A for loop", "B": "Mechanism handling async operations in a single thread", "C": "A testing tool", "D": "A database connection pool"}, "correct": "B"},
                {"question": "What is rate limiting?", "options": {"A": "Slowing down code", "B": "Controlling how many requests a client can make", "C": "A database optimisation", "D": "A UI animation"}, "correct": "B"},
                {"question": "What is optimistic vs pessimistic locking?", "options": {"A": "Security settings", "B": "Strategies for handling concurrent data access", "C": "UI patterns", "D": "Deployment strategies"}, "correct": "B"},
            ],
        },
        "ui/ux designer": {
            "basic": [
                {"question": "What does UX stand for?", "options": {"A": "User Experience", "B": "Unified Exchange", "C": "User Extension", "D": "Usability Export"}, "correct": "A"},
                {"question": "What does UI stand for?", "options": {"A": "User Interface", "B": "Unified Integration", "C": "User Interaction", "D": "Usability Index"}, "correct": "A"},
                {"question": "What is a wireframe?", "options": {"A": "A finished design", "B": "A low-fidelity layout sketch", "C": "A CSS framework", "D": "A font system"}, "correct": "B"},
                {"question": "What is a prototype?", "options": {"A": "Final product", "B": "Interactive model of a design", "C": "A colour scheme", "D": "A font choice"}, "correct": "B"},
                {"question": "What is Figma?", "options": {"A": "A programming language", "B": "A UI design tool", "C": "A database", "D": "A testing tool"}, "correct": "B"},
                {"question": "What is user research?", "options": {"A": "Searching for users online", "B": "Understanding user needs and behaviours", "C": "Analysing competitor products", "D": "Testing code"}, "correct": "B"},
                {"question": "What is a call to action (CTA)?", "options": {"A": "A phone number", "B": "A button prompting user action", "C": "A page title", "D": "A menu item"}, "correct": "B"},
                {"question": "What is white space in design?", "options": {"A": "White coloured elements", "B": "Empty space between elements", "C": "The background colour", "D": "Blank pages"}, "correct": "B"},
                {"question": "What is a persona in UX?", "options": {"A": "A user account", "B": "A fictional representation of a target user", "C": "A design style", "D": "A font choice"}, "correct": "B"},
                {"question": "What is usability testing?", "options": {"A": "Testing code", "B": "Testing a design with real users", "C": "Checking colours", "D": "Reviewing fonts"}, "correct": "B"},
            ],
            "intermediate": [
                {"question": "What is the Gestalt principle of proximity?", "options": {"A": "Similar items look related", "B": "Items close together appear related", "C": "Symmetrical designs are preferred", "D": "Figures stand out from backgrounds"}, "correct": "B"},
                {"question": "What is an affinity diagram?", "options": {"A": "A chart type", "B": "Organising ideas into groups", "C": "A wireframe tool", "D": "A colour system"}, "correct": "B"},
                {"question": "What is a user journey map?", "options": {"A": "A geographical map", "B": "Visual representation of user's experience", "C": "A sitemap", "D": "A navigation menu"}, "correct": "B"},
                {"question": "What is accessibility in design?", "options": {"A": "Fast loading", "B": "Making design usable by people with disabilities", "C": "Mobile responsiveness", "D": "Colourful design"}, "correct": "B"},
                {"question": "What is a design system?", "options": {"A": "A computer system", "B": "Collection of reusable design components and guidelines", "C": "A project management tool", "D": "A testing framework"}, "correct": "B"},
                {"question": "What is information architecture?", "options": {"A": "Building design", "B": "Organising and structuring content", "C": "A font system", "D": "A coding practice"}, "correct": "B"},
                {"question": "What is a heuristic evaluation?", "options": {"A": "A user test", "B": "Expert review of UI against usability principles", "C": "A colour analysis", "D": "A performance test"}, "correct": "B"},
                {"question": "What is progressive disclosure?", "options": {"A": "Revealing all info at once", "B": "Showing information gradually as needed", "C": "A loading animation", "D": "A font technique"}, "correct": "B"},
                {"question": "What is the F-pattern in web reading?", "options": {"A": "A font style", "B": "How users scan web pages in an F-shape", "C": "A grid system", "D": "A navigation pattern"}, "correct": "B"},
                {"question": "What is cognitive load?", "options": {"A": "Server load", "B": "Mental effort required to use an interface", "C": "Page loading time", "D": "Database query time"}, "correct": "B"},
            ],
            "critical": [
                {"question": "What is the difference between qualitative and quantitative research?", "options": {"A": "They are the same", "B": "Qualitative = insights/opinions; quantitative = numbers/metrics", "C": "Quantitative comes first", "D": "Qualitative uses numbers"}, "correct": "B"},
                {"question": "What is dark UI pattern?", "options": {"A": "A dark-themed design", "B": "Design that tricks users into unintended actions", "C": "A night mode", "D": "Low contrast design"}, "correct": "B"},
                {"question": "What is the Jobs-to-be-Done framework?", "options": {"A": "A job listing tool", "B": "Understanding what users are trying to achieve", "C": "A project management method", "D": "A coding framework"}, "correct": "B"},
                {"question": "What is emotional design?", "options": {"A": "Designing for emotional users", "B": "Creating experiences that evoke specific emotions", "C": "Using emotional colours", "D": "Avoiding negative emotions"}, "correct": "B"},
                {"question": "What is the difference between usability and desirability?", "options": {"A": "They are the same", "B": "Usability = ease of use; desirability = emotional appeal", "C": "Desirability is more important", "D": "Usability includes desirability"}, "correct": "B"},
            ],
        },
    }

    role_bank = bank.get(role, bank["software developer"])
    level_bank = role_bank.get(level, role_bank["basic"])
    return level_bank[:count]


def get_fallback_descriptive(role):
    fallbacks = {
        "software developer": [
            {"question": "Write a Python function that takes a list of numbers and returns only the even numbers sorted in descending order. Explain your logic."},
            {"question": "Write a function to check if a string is a palindrome. Show your solution and explain how it works."},
            {"question": "Write a simple class in Python with a constructor and one method. Explain what each part does."},
        ],
        "full stack developer": [
            {"question": "Write a simple Express.js route that accepts a POST request with a name field and returns a JSON greeting. Explain each line."},
            {"question": "Write an SQL query to find all users who registered in the last 30 days from a 'users' table with a 'created_at' column."},
            {"question": "Write a React functional component that displays a counter with increment and decrement buttons using useState."},
        ],
        "web developer": [
            {"question": "Write the HTML and CSS for a simple card component with a title, description, and a button. Explain your structure."},
            {"question": "Write a JavaScript function that fetches data from an API URL and displays the results. Handle errors appropriately."},
            {"question": "Write CSS to center a div both horizontally and vertically using Flexbox. Explain each property used."},
        ],
        "data analyst": [
            {"question": "Write a SQL query to find the top 5 products by total sales from a table with columns: product_name, quantity, price."},
            {"question": "Write Python code using Pandas to read a CSV, remove rows with missing values, and show the first 5 rows."},
            {"question": "Describe how you would create a dashboard to track monthly sales. What charts would you use and why?"},
        ],
        "cyber security": [
            {"question": "Write pseudocode or Python to demonstrate how SQL injection works and how to prevent it using parameterised queries."},
            {"question": "Describe a secure login system. What security measures would you implement and why?"},
            {"question": "Write a simple Python script to check if a password meets security requirements (length, uppercase, numbers, symbols)."},
        ],
        "ui/ux designer": [
            {"question": "Describe the wireframe for a mobile login screen. List all elements and explain your UX decisions."},
            {"question": "A user complains the checkout process has too many steps. Describe how you would redesign it with better UX."},
            {"question": "Design a colour palette and typography system for a health app. Explain why you chose each element."},
        ],
    }
    return fallbacks.get(role, fallbacks["software developer"])


# ─────────────────────────────────────────────
# SCORE HELPERS
# ─────────────────────────────────────────────

def score_mcq_level(answers_dict, questions, time_taken):
    total   = len(questions)
    correct = 0
    review  = []

    for i, q in enumerate(questions):
        user_ans   = answers_dict.get(str(i))
        is_correct = user_ans == q.get("correct")
        if is_correct:
            correct += 1
        review.append({
            "question":         q.get("question", ""),
            "user_answer":      user_ans,
            "user_answer_text": q.get("options", {}).get(user_ans, "Skipped") if user_ans else "Skipped",
            "correct_answer":   q.get("correct", ""),
            "correct_text":     q.get("options", {}).get(q.get("correct", ""), ""),  # ← fixed key name
            "correct":          is_correct,
            "skipped":          user_ans is None,
        })

    pct      = round((correct / total) * 100) if total > 0 else 0
    avg_time = round(time_taken / total, 1)    if total > 0 else 0
    passed   = pct >= 60

    if avg_time <= 15:   speed = 100
    elif avg_time <= 30: speed = 75
    elif avg_time <= 45: speed = 50
    else:                speed = 25

    return {
        "correct": correct, "total": total, "pct": pct,
        "passed": passed, "avg_time": avg_time, "speed": speed,
        "review": review,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. UPDATED INTERVIEW QUESTION GENERATOR  (replaces ai_generate_question)
#    This version generates structured questions per round type
# ─────────────────────────────────────────────────────────────────────────────

ROUND_TYPES = {
    1: "self_introduction",
    2: "technical",
    3: "behavioural",
    4: "scenario",
}

ROUND_PROMPTS = {
    "self_introduction": """You are starting a job interview for a {role} position.
Ask the candidate to introduce themselves.
The question must:
- Ask them to introduce themselves professionally
- Prompt them to mention their background, technical skills in {role}, key projects/experience, and motivation for this role
- Be warm but professional in tone
- Be ONE question only (2-3 sentences max)
Respond with ONLY the question.""",

    "technical": """You are interviewing a {role} candidate. This is a technical interview question.
{history}
Ask ONE specific, practical technical question relevant to {role} at {difficulty} level.
- Focus on a core skill or concept for {role}
- It should require real technical knowledge to answer well
- Avoid yes/no questions — ask them to explain, describe, or demonstrate
- 1-2 sentences max
Respond with ONLY the question.""",

    "behavioural": """You are interviewing a {role} candidate. This is a behavioural interview question.
{history}
Ask ONE behavioural question using the STAR format trigger for {role}.
- Use phrases like "Tell me about a time when...", "Describe a situation where...", "Give me an example of..."
- Focus on: teamwork, problem-solving, conflict resolution, meeting deadlines, or handling failure
- Make it relevant to {role} work context
- 1-2 sentences max
Respond with ONLY the question.""",

    "scenario": """You are interviewing a {role} candidate. This is a scenario/situational question.
{history}
Ask ONE scenario question: "What would you do if..." or "How would you handle..."
- Present a realistic workplace situation relevant to {role}
- The scenario should have some complexity or trade-off to reason through
- 2-3 sentences max
Respond with ONLY the question.""",
}

INTERVIEW_FALLBACKS = {
    "software developer": {
        "self_introduction": "Please introduce yourself. Tell me about your background, your strongest technical skills as a software developer, any notable projects you've worked on, and what draws you to this role.",
        "technical":        "Explain the difference between a stack and a queue. Give a real-world example where you would choose one over the other.",
        "behavioural":      "Tell me about a time you had to debug a particularly tricky issue. How did you approach it, and what did you learn from the experience?",
        "scenario":         "Imagine you're halfway through a sprint and you discover that a core feature you've built won't work due to a dependency your team didn't know about. How would you handle this?",
    },
    "full stack developer": {
        "self_introduction": "Please introduce yourself. Walk me through your full-stack experience — the frontend and backend technologies you're most confident in, a project you're proud of, and why you want to work as a full stack developer.",
        "technical":        "Explain how you would design a secure login system from scratch. Cover both the frontend and backend, and mention at least two security measures.",
        "behavioural":      "Describe a time when you had to balance frontend and backend work simultaneously under a tight deadline. How did you prioritise?",
        "scenario":         "Your team's React app is loading very slowly on mobile — FCP is 8 seconds. The backend APIs are all under 200ms. How would you diagnose and fix the performance issue?",
    },
    "web developer": {
        "self_introduction": "Please introduce yourself. Tell me about your experience with HTML, CSS, and JavaScript, the types of websites you've built, and what motivates you as a web developer.",
        "technical":        "What is the CSS box model? Explain each layer and describe a common bug caused by misunderstanding it.",
        "behavioural":      "Tell me about a time you received critical design feedback from a client after you'd already built the page. How did you handle it?",
        "scenario":         "A client wants their website to load in under 2 seconds but the current load time is 7 seconds. What steps would you take to optimise it?",
    },
    "data analyst": {
        "self_introduction": "Please introduce yourself. Tell me about your experience with data analysis, the tools you use most (SQL, Python, Excel, etc.), a project where your analysis drove a business decision, and what you enjoy most about working with data.",
        "technical":        "Explain the difference between INNER JOIN, LEFT JOIN, and FULL OUTER JOIN in SQL. When would you use each?",
        "behavioural":      "Tell me about a time your analysis uncovered something unexpected. How did you validate it was real and not a data error?",
        "scenario":         "Your manager asks you to analyse why sales dropped 30% last quarter. You have access to sales, marketing, and customer data. Walk me through your approach from start to finish.",
    },
    "cyber security": {
        "self_introduction": "Please introduce yourself. Tell me about your background in cybersecurity, the areas you specialise in (networking, pen testing, incident response, etc.), any certifications or notable projects, and what draws you to this field.",
        "technical":        "Explain what SQL injection is, how it works, and describe three ways to prevent it in a web application.",
        "behavioural":      "Tell me about a time you had to explain a serious security risk to a non-technical stakeholder. How did you communicate the urgency without causing panic?",
        "scenario":         "At 3am you receive an alert that an internal server is sending unusual outbound traffic to an unknown IP. No one else is available. Walk me through your step-by-step incident response.",
    },
    "ui/ux designer": {
        "self_introduction": "Please introduce yourself. Tell me about your design background, the tools you work with (Figma, etc.), a project you're most proud of, and what your design philosophy is.",
        "technical":        "Walk me through your design process from receiving a brief to delivering a final design. What steps do you never skip and why?",
        "behavioural":      "Tell me about a time when user research revealed your design assumptions were wrong. How did you respond and what changed?",
        "scenario":         "A product manager wants to add 5 new features to the homepage in the next release. You believe this will hurt the user experience significantly. How do you handle this disagreement?",
    },
}


def ai_generate_question(role, interview_history=None, question_number=1, assessment_score=None):
    round_type = ROUND_TYPES.get(question_number, "scenario")

    history_text = ""
    if interview_history:
        history_text = "Previous Q&A:\n"
        for i, qa in enumerate(interview_history, 1):
            history_text += f"Q{i}: {qa['question']}\nA{i}: {qa['answer'][:200]}...\n"

    difficulty = "intermediate"
    if assessment_score is not None:
        if assessment_score >= 75:  difficulty = "advanced"
        elif assessment_score < 40: difficulty = "beginner"

    prompt_template = ROUND_PROMPTS.get(round_type, ROUND_PROMPTS["technical"])
    prompt = prompt_template.format(
        role=role,
        history=history_text,
        difficulty=difficulty,
    )

    result = call_claude(prompt, max_tokens=200)
    if result:
        return result.strip()

    # Fallback
    role_fallbacks = INTERVIEW_FALLBACKS.get(role, INTERVIEW_FALLBACKS["software developer"])
    return role_fallbacks.get(round_type, f"Tell me about your experience with {role}.")



# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
from flask_session import Session
import tempfile

app.config["SESSION_TYPE"]             = "filesystem"
import os
app.config["SESSION_FILE_DIR"]          = os.path.join(os.path.dirname(__file__), "flask_sessions")
app.config["SESSION_FILE_THRESHOLD"]    = 100
app.config["SESSION_PERMANENT"]         = False
app.config["SESSION_COOKIE_SAMESITE"]   = "Lax"
Session(app)
app.secret_key = os.environ.get("SECRET_KEY", "career_ai_secret_change_in_prod")


def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/")
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    os.makedirs("uploads", exist_ok=True)
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, password TEXT NOT NULL
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL, role TEXT, overall REAL,
        rating TEXT, job_ready TEXT, summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email    = request.form["email"].strip()
        password = request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["name"]    = user["name"]
            return redirect("/dashboard")
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name     = request.form["name"].strip()
        email    = request.form["email"].strip()
        password = generate_password_hash(request.form["password"])
        try:
            conn = get_db()
            conn.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)", (name,email,password))
            conn.commit()
            conn.close()
            flash("Account created! Please log in.", "success")
            return redirect("/")
        except sqlite3.IntegrityError:
            flash("Email already registered.", "error")
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    conn    = get_db()
    history = conn.execute(
        "SELECT role,overall,rating,job_ready,created_at FROM results WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", name=session["name"], history=history)


# ─────────────────────────────────────────────
# RESUME
# ─────────────────────────────────────────────

@app.route("/resume")
@login_required
def resume():
    return render_template("resume.html")


@app.route("/analyze_resume", methods=["POST"])
@login_required
def analyze_resume():
    file = request.files.get("resume")
    if not file or file.filename == "":
        flash("Please upload a PDF file.", "error")
        return redirect("/resume")

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                pt = page.extract_text()
                if pt: text += pt
    except Exception:
        flash("Could not read PDF.", "error")
        return redirect("/resume")
    finally:
        if os.path.exists(filepath): os.remove(filepath)

    detected = detect_skills(text)
    career   = recommend_career(detected)
    missing  = missing_skills(career, detected)
    return render_template("resume_result.html", detected_skills=detected, career=career, missing_skills=missing)


# ─────────────────────────────────────────────
# CAREER SELECTION
# ─────────────────────────────────────────────

@app.route("/career", methods=["GET","POST"])
@login_required
def career():
    if request.method == "POST":
        role = request.form["role"].strip().lower()
        if role not in ROLES:
            flash("Please select a valid career.", "error")
            return redirect("/career")

        session["role"]              = role
        session["interview_history"] = []
        session["interview_round"]   = 1
        session["current_question"]  = None
        session["assessment_data"]   = None
        session["mcq_questions"]     = None

        return redirect("/assessment")

    return render_template("career.html", roles=ROLES)


# ─────────────────────────────────────────────
# ASSESSMENT — GET: render, POST: score
# ─────────────────────────────────────────────

@app.route("/assessment", methods=["GET","POST"])
@login_required
def assessment():
    if "role" not in session:
        return redirect("/career")

    role = session["role"]

    if request.method == "POST":
        # Parse answers
        def parse_answers(key):
            raw = request.form.get(key, "{}")
            try:    return json.loads(raw)
            except: return {}

        ans_l1   = parse_answers("answers_l1")
        ans_l2   = parse_answers("answers_l2")
        ans_l3   = parse_answers("answers_l3")
        ans_desc = parse_answers("answers_desc")

        time_l1   = int(request.form.get("time_taken_l1",   300))
        time_l2   = int(request.form.get("time_taken_l2",   300))
        time_l3   = int(request.form.get("time_taken_l3",   180))
        time_desc = int(request.form.get("time_taken_desc", 900))

        questions = session.get("mcq_questions", {})

        # Score each level
        score_l1 = score_mcq_level(ans_l1, questions.get("l1", []), time_l1)
        score_l2 = score_mcq_level(ans_l2, questions.get("l2", []), time_l2)
        score_l3 = score_mcq_level(ans_l3, questions.get("l3", []), time_l3)

        # Score descriptive
        desc_results = []
        desc_qs      = questions.get("desc", [])
        for i, q in enumerate(desc_qs):
            ans     = ans_desc.get(str(i), "")
            sc, fb  = ai_score_descriptive(role, q.get("question",""), ans)
            desc_results.append({"question": q.get("question",""), "answer": ans, "score": sc, "feedback": fb})

        desc_avg = round(sum(r["score"] for r in desc_results) / len(desc_results), 1) if desc_results else 5

        assessment_data = {
            "l1":        score_l1,
            "l2":        score_l2,
            "l3":        score_l3,
            "desc":      desc_results,
            "desc_avg":  desc_avg,
            "overall_score": round((score_l1["pct"] + score_l2["pct"] + score_l3["pct"]) / 3),
        }

        session["assessment_data"] = assessment_data
        return redirect("/assessment_result")

    # GET — generate questions
    questions = session.get("mcq_questions")
    if not questions:
        questions = {
            "l1":   ai_generate_mcq(role, "basic",        10),
            "l2":   ai_generate_mcq(role, "intermediate", 10),
            "l3":   ai_generate_mcq(role, "critical",      5),
            "desc": ai_generate_descriptive(role,           3),
        }
        session["mcq_questions"] = questions

    return render_template(
        "assessment.html",
        role=role,
        questions_l1=questions["l1"],
        questions_l2=questions["l2"],
        questions_l3=questions["l3"],
        questions_desc=questions["desc"],
    )


# ─────────────────────────────────────────────
# REVISION NOTES ENDPOINT (AJAX)
# ─────────────────────────────────────────────

@app.route("/get_revision_notes", methods=["POST"])
@login_required
def get_revision_notes():
    data    = request.get_json()
    role    = data.get("role", session.get("role", ""))
    level   = data.get("level", "")
    correct = data.get("correct", 0)
    total   = data.get("total", 10)
    notes   = ai_revision_notes(role, level, correct, total)
    return jsonify({"notes": notes})


# ─────────────────────────────────────────────
# ASSESSMENT RESULT
# ─────────────────────────────────────────────

@app.route("/assessment_result")
@login_required
def assessment_result():
    if "assessment_data" not in session:
        return redirect("/assessment")

    data = session["assessment_data"]
    role = session.get("role", "")

    return render_template(
        "assessment_result.html",
        role=role,
        score_l1=data.get("l1", {}),
        score_l2=data.get("l2", {}),
        score_l3=data.get("l3", {}),
        desc_results=data.get("desc", []),
        desc_avg=data.get("desc_avg", 0),
        overall=data.get("overall_score", 0),
    )

# ─────────────────────────────────────────────
# INTERVIEW
# ─────────────────────────────────────────────

@app.route("/interview", methods=["GET", "POST"])
@login_required
def interview():
    if "role" not in session:
        return redirect("/career")

    role             = session["role"]
    history          = session.get("interview_history", [])
    round_n          = session.get("interview_round", 1)
    assessment_data  = session.get("assessment_data", {})
    assessment_score = assessment_data.get("overall_score", 50) if assessment_data else 50
    total_rounds     = 4

    if round_n > total_rounds:
        return redirect("/simulation")

    if request.method == "POST":
        answer   = request.form.get("answer", "").strip()
        question = session.get("current_question", "")
        score, feedback = ai_score_answer(role, question, answer)

        history.append({
            "question":   question,
            "answer":     answer,
            "score":      score,
            "feedback":   feedback,
            "round":      round_n,
            "round_type": ROUND_TYPES.get(round_n, "technical"),
        })
        session["interview_history"] = history
        session["interview_round"]   = round_n + 1

        if round_n >= total_rounds:
            return redirect("/simulation")

        next_q = ai_generate_question(role, history, round_n + 1, assessment_score)
        session["current_question"] = next_q

        return render_template(
            "interview.html",
            role=role,
            question=next_q,
            round_number=round_n + 1,
            total_rounds=total_rounds,
            last_score=score,
            last_feedback=feedback,
        )

    if not session.get("current_question"):
        q = ai_generate_question(role, [], 1, assessment_score)
        session["current_question"] = q
    else:
        q = session["current_question"]

    return render_template(
        "interview.html",
        role=role,
        question=q,
        round_number=round_n,
        total_rounds=total_rounds,
        last_score=None,
        last_feedback=None,
    )



# ─────────────────────────────────────────────────────────────────────────────
# 1. REAL-WORLD SIMULATION TASKS  (replaces ai_generate_simulation)
#    Add this dictionary above the simulation route
# ─────────────────────────────────────────────────────────────────────────────

SIMULATION_TASKS = {
    "software developer": {
        "task_type_label":  "Code Review + Fix",
        "task_type_class":  "type-debug",
        "task_title":       "Bug Hunt: Fix a Broken REST API Service",
        "task_context":     "You've just joined a startup as a software developer. Your team lead has flagged a production bug — the user registration endpoint is failing silently. Users report they can register but their data never appears in the database.",
        "task_background":  "The app is a Python Flask REST API with SQLite. It has been live for 3 months. The bug was introduced in the last deployment when a junior developer refactored the database layer.",
        "task_your_role":   "You are the on-call developer. You must identify the bug, fix it, write a test for it, and document what went wrong.",
        "task_constraints": "No external libraries beyond Flask and SQLite. Fix must be backward-compatible. You cannot break existing endpoints.",
        "task_parts": [
            {
                "title": "Review the broken code below and identify ALL bugs",
                "detail": "List each bug you find with its line number, what it does wrong, and why it causes the silent failure.",
                "code": """@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data['name']
    email = data['email']
    password = data['password']
    
    conn = sqlite3.connect('users.db')
    conn.execute(
        "INSERT INTO users VALUES (?, ?, ?)",
        (name, email, password)
    )
    conn.close()   # Bug is here — what's missing?
    
    return jsonify({'message': 'User created'}), 200""",
                "code_lang": "python",
            },
            {
                "title": "Write the corrected version of the function",
                "detail": "Rewrite the entire register() function with all bugs fixed. Add proper error handling, password hashing, and input validation.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Write a unit test for the fixed function",
                "detail": "Write a simple test that verifies: (a) a valid registration succeeds and (b) a duplicate email returns an error.",
                "code": None,
                "code_lang": "",
            },
        ],
        "task_deliverables": [
            "List every bug found with explanation",
            "Corrected register() function with proper error handling",
            "Password hashing using werkzeug or hashlib",
            "Unit test covering success and duplicate-email cases",
            "Brief explanation of what caused the silent failure",
        ],
        "task_scoring": "Scored on: bug identification accuracy (30%), fix correctness (40%), test quality (20%), explanation clarity (10%).",
    },

    "full stack developer": {
        "task_type_label":  "System Design + Code",
        "task_type_class":  "type-plan",
        "task_title":       "Design & Build a Real-Time Notification System",
        "task_context":     "You work at a mid-sized SaaS company. Product has requested a real-time notification bell (like LinkedIn's) for their web app. You have one sprint (5 days) to design and implement it.",
        "task_background":  "Current stack: React frontend, Node.js/Express backend, PostgreSQL database. The app has 50,000 active users. Notifications should appear within 2 seconds of an event.",
        "task_your_role":   "You are the sole full-stack developer on this feature. You must design the architecture, implement the backend API and frontend component, and consider scalability.",
        "task_constraints": "No paid third-party notification services. Must work on mobile. Must handle 1,000 concurrent users without crashing.",
        "task_parts": [
            {
                "title": "Design the database schema for notifications",
                "detail": "Design a PostgreSQL table (or tables) to store notifications. Consider: user targeting, read/unread state, notification types, timestamps, and soft deletion.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Design the backend API endpoints",
                "detail": "List all REST API endpoints needed. For each, specify: HTTP method, URL, request body, response format, and auth requirements.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Choose and justify a real-time delivery mechanism",
                "detail": "Compare WebSockets vs Server-Sent Events (SSE) vs Polling for this use case. Choose one and explain why it fits best given the constraints.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Write the React notification bell component",
                "detail": "Write a functional React component that: shows an unread count badge, connects to your chosen real-time endpoint, marks notifications as read on click, and displays a dropdown list.",
                "code": None,
                "code_lang": "",
            },
        ],
        "task_deliverables": [
            "SQL schema with CREATE TABLE statements",
            "Complete API endpoint specification",
            "Real-time mechanism choice with comparison and justification",
            "React NotificationBell component with hooks",
            "Brief note on how you'd handle 1,000 concurrent connections",
        ],
        "task_scoring": "Scored on: schema design (25%), API design (25%), real-time choice reasoning (20%), React component quality (30%).",
    },

    "web developer": {
        "task_type_label":  "Build + Optimise",
        "task_type_class":  "type-code",
        "task_title":       "Build a Responsive Product Card with Accessibility",
        "task_context":     "You're a web developer at an e-commerce agency. A client has sent you a Figma design for a product card component. You need to build it, make it fully responsive, and ensure it passes WCAG 2.1 AA accessibility standards.",
        "task_background":  "The client sells electronics. Their site gets 60% mobile traffic. Their last audit failed on colour contrast, missing alt text, and keyboard navigation. The new component must fix all of this.",
        "task_your_role":   "You are building this component in isolation. It will be dropped into multiple pages. It must be self-contained HTML/CSS/JS with no external dependencies.",
        "task_constraints": "No CSS frameworks (no Bootstrap/Tailwind). Pure HTML, CSS, JavaScript only. Must work on IE11 equivalent (no CSS Grid, use Flexbox). Loading time under 50ms.",
        "task_parts": [
            {
                "title": "Write the complete HTML structure for the product card",
                "detail": "Include: product image, name, price, rating (stars), 'Add to Cart' button, and a 'Wishlist' toggle. Use semantic HTML5 elements and all required ARIA attributes.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Write the CSS for the card (responsive, no frameworks)",
                "detail": "Style the card to look professional. Must be responsive for mobile (320px) through desktop (1440px). Use CSS custom properties for colours. Ensure 4.5:1 contrast ratio.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Identify and fix all accessibility issues in this snippet",
                "detail": "List every WCAG failure below and provide the corrected HTML.",
                "code": """<div class="product" onclick="addToCart()">
  <img src="phone.jpg">
  <div class="name" style="color: #aaa">iPhone 15</div>
  <div class="price">₹79,999</div>
  <div class="btn" onclick="addToCart()">Buy</div>
  <span class="heart" onclick="wishlist()">♥</span>
</div>""",
                "code_lang": "html",
            },
        ],
        "task_deliverables": [
            "Complete semantic HTML with all ARIA attributes",
            "Responsive CSS using Flexbox and CSS custom properties",
            "List of all accessibility violations found with fixes",
            "Explanation of how keyboard navigation works in your component",
        ],
        "task_scoring": "Scored on: HTML semantics (25%), CSS quality & responsiveness (30%), accessibility fixes (30%), keyboard nav explanation (15%).",
    },

    "data analyst": {
        "task_type_label":  "Data Analysis",
        "task_type_class":  "type-data",
        "task_title":       "Investigate a Sales Drop: Root Cause Analysis",
        "task_context":     "You work as a data analyst at an online retailer. The head of sales called an urgent meeting — revenue dropped 23% last month compared to the same month last year. Your job: find out why.",
        "task_background":  "The company sells across 4 categories: Electronics, Clothing, Home & Garden, Sports. They operate in 3 regions: North, South, West. They have a loyalty programme. Last month there was a website redesign.",
        "task_your_role":   "You have 2 hours before the board meeting. You must present: what happened, why it happened, and what to do next.",
        "task_constraints": "You only have summary-level data (no raw rows). You must make logical inferences. You cannot request more data before the meeting.",
        "task_parts": [
            {
                "title": "Analyse this data table and find the root cause",
                "detail": "Study the numbers below carefully. Identify which category, region, or segment drove the decline and explain your reasoning.",
                "code": """Category      | This Month | Last Year | Change
Electronics   |  ₹4.2L     |  ₹4.1L    | +2.4%
Clothing      |  ₹1.8L     |  ₹3.9L    | -53.8%  ← 
Home & Garden |  ₹2.1L     |  ₹2.0L    | +5.0%
Sports        |  ₹0.9L     |  ₹0.8L    | +12.5%

Region        | This Month | Last Year | Change
North         |  ₹3.8L     |  ₹3.7L    | +2.7%
South         |  ₹1.2L     |  ₹3.2L    | -62.5%  ←
West          |  ₹4.0L     |  ₹3.9L    | +2.6%

Loyalty Members:    Visits -5%,  Conversion -3%
Non-Members:        Visits -8%,  Conversion -41%  ←
Website Redesign:   Launched on the 3rd of last month""",
                "code_lang": "text",
            },
            {
                "title": "Write the SQL query to verify your hypothesis",
                "detail": "Write SQL to extract the data you would need to confirm your root cause theory. Assume tables: orders(id, date, category, region, customer_id, revenue), customers(id, is_loyalty_member, signup_date).",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Create your board presentation structure",
                "detail": "Outline a 5-slide deck for the board meeting. For each slide: write the title, the key message, and what chart/visual you'd use.",
                "code": None,
                "code_lang": "",
            },
        ],
        "task_deliverables": [
            "Root cause identification with data-backed reasoning",
            "SQL query to validate hypothesis",
            "3 actionable recommendations with expected impact",
            "5-slide board deck outline",
            "Risk: what else could explain the drop (alternative hypotheses)",
        ],
        "task_scoring": "Scored on: analysis accuracy (35%), SQL correctness (25%), recommendations quality (25%), communication clarity (15%).",
    },

    "cyber security": {
        "task_type_label":  "Incident Response",
        "task_type_class":  "type-debug",
        "task_title":       "Respond to a Live Security Breach",
        "task_context":     "You are a security analyst at a fintech company. At 2:47 AM, your SIEM fires a P1 alert. The logs show unusual activity on the customer database server. Your incident response plan says you have 15 minutes to make a containment decision.",
        "task_background":  "Company handles 200,000 customer payment records. Compliance requires breach notification within 72 hours. The affected server runs the customer-facing API. Downtime costs ₹50,000/minute.",
        "task_your_role":   "You are the on-call security analyst. No one else is awake. You must triage, contain, investigate, and begin the incident report.",
        "task_constraints": "You cannot take down the server without VP approval (who is unreachable). You have read-only access to logs. You can block IPs at the firewall level.",
        "task_parts": [
            {
                "title": "Triage these log entries — what's happening?",
                "detail": "Analyse the logs below. Identify the attack type, attacker behaviour, and what data may have been accessed.",
                "code": """02:31:14 | IP: 185.220.101.47 | GET /api/users?id=1 | 200
02:31:15 | IP: 185.220.101.47 | GET /api/users?id=2 | 200
02:31:15 | IP: 185.220.101.47 | GET /api/users?id=3 | 200
[... 847 similar requests in 4 minutes ...]
02:35:22 | IP: 185.220.101.47 | GET /api/users?id=848 | 200
02:35:23 | IP: 185.220.101.47 | GET /api/users?id=1' OR '1'='1 | 500
02:35:24 | IP: 185.220.101.47 | GET /api/users?id=1 UNION SELECT * FROM users-- | 200
02:35:25 | DB  | QUERY: SELECT * FROM users WHERE id=1 UNION SELECT * FROM users--
02:35:25 | DB  | Rows returned: 200000
02:47:01 | SIEM| ALERT P1: Bulk data exfiltration detected (200K rows, 185.220.101.47)""",
                "code_lang": "text",
            },
            {
                "title": "Write your immediate containment actions (prioritised list)",
                "detail": "List exactly what you do in the next 15 minutes. For each action: what you do, why, and what risk it carries.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Write the vulnerable API endpoint and its secure fix",
                "detail": "Based on the logs, write what the vulnerable code probably looks like, then write the secure version using parameterised queries.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Begin the incident report",
                "detail": "Write the first section of a formal incident report: Timeline, Attack Type, Data Affected, Immediate Actions Taken, and Regulatory Notification requirements.",
                "code": None,
                "code_lang": "",
            },
        ],
        "task_deliverables": [
            "Attack identification: type, technique (OWASP category), severity",
            "Prioritised containment action list with reasoning",
            "Vulnerable code and secure parameterised fix",
            "Partial incident report (timeline + impact assessment)",
            "Long-term remediation recommendations (minimum 3)",
        ],
        "task_scoring": "Scored on: attack identification (25%), containment decisions (30%), code fix correctness (25%), incident report quality (20%).",
    },

    "ui/ux designer": {
        "task_type_label":  "UX Redesign",
        "task_type_class":  "type-design",
        "task_title":       "Redesign a Failing Checkout Flow",
        "task_context":     "You are a UX designer at an e-commerce company. The checkout page has a 71% abandonment rate — one of the worst in the industry. User interviews revealed: 'too many steps', 'confusing', 'I don't trust it'. Your job: redesign it.",
        "task_background":  "Current flow has 6 steps: Cart → Login/Register → Shipping → Payment → Review → Confirmation. Average completion time is 8 minutes. Mobile accounts for 70% of traffic. 40% of drop-offs happen at the Login step.",
        "task_your_role":   "You are the sole UX designer. You must redesign the flow, justify every decision with UX principles, and present it to the stakeholders.",
        "task_constraints": "Cannot remove login entirely (business requirement). Must keep existing payment gateway UI. Must work for first-time and returning users. Redesign must be achievable in one 2-week sprint.",
        "task_parts": [
            {
                "title": "Identify every UX problem in the current flow",
                "detail": "List every UX issue, mapping each to a specific UX principle it violates (e.g. Hick's Law, Fitts' Law, Miller's Law, Nielsen's Heuristics). Explain why each causes abandonment.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Design your new checkout flow",
                "detail": "Describe each screen in your redesigned flow. For each screen: name, purpose, key elements, micro-interactions, and what UX principle guides the design. Aim to reduce to 3 steps or fewer.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Design the Login step to reduce 40% drop-off",
                "detail": "This is the highest abandonment point. Describe in detail how you'd redesign this specific step. Include: layout, copy, social login placement, guest checkout option, trust signals, and error states.",
                "code": None,
                "code_lang": "",
            },
            {
                "title": "Define your success metrics and test plan",
                "detail": "How will you know if your redesign worked? Define 3 KPIs with target values, describe how you'd A/B test it, and list 5 usability test tasks you'd give to participants.",
                "code": None,
                "code_lang": "",
            },
        ],
        "task_deliverables": [
            "Problem list with UX principle violations mapped",
            "New flow: screen-by-screen description (3 steps or fewer)",
            "Detailed login step redesign with all states",
            "3 KPIs with targets and measurement method",
            "A/B test plan and usability test tasks",
        ],
        "task_scoring": "Scored on: problem identification depth (25%), redesign logic (35%), login redesign detail (25%), test plan quality (15%).",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. UPDATED SIMULATION ROUTE  (replaces the existing /simulation route)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/simulation", methods=["GET", "POST"])
@login_required
def simulation():
    if "role" not in session:
        return redirect("/career")

    role            = session["role"]
    history         = session.get("interview_history", [])
    assessment_data = session.get("assessment_data", {})
    assessment_score = assessment_data.get("overall_score", 50) if assessment_data else 50

    if request.method == "POST":
        answer   = request.form.get("answer", "").strip()
        task_obj = session.get("simulation_task_obj", {})
        task_str = task_obj.get("task_title", "") + " — " + task_obj.get("task_context", "")
        score, feedback = ai_score_answer(role, task_str, answer)
        session["simulation_score"]    = score
        session["simulation_feedback"] = feedback
        session["simulation_answer"]   = answer
        return redirect("/result")

    # Pick task for this role
    task = SIMULATION_TASKS.get(role, SIMULATION_TASKS["software developer"])
    session["simulation_task_obj"] = task

    # Build parts list (Jinja2 needs dicts)
    return render_template(
        "simulation.html",
        role=role,
        task_type_label  = task["task_type_label"],
        task_type_class  = task["task_type_class"],
        task_title       = task["task_title"],
        task_context     = task["task_context"],
        task_background  = task["task_background"],
        task_your_role   = task["task_your_role"],
        task_constraints = task["task_constraints"],
        task_parts       = task["task_parts"],
        task_deliverables= task["task_deliverables"],
        task_scoring     = task["task_scoring"],
    )


@app.route("/result")
@login_required
def result():
    if "role" not in session:
        return redirect("/career")

    role             = session.get("role","")
    history          = session.get("interview_history",[])
    simulation_score = session.get("simulation_score", 5)
    sim_feedback     = session.get("simulation_feedback","")
    simulation_task  = session.get("simulation_task","")
    assessment_data  = session.get("assessment_data", {"l1":{},"l2":{},"l3":{},"overall_score":0})

    report = ai_generate_report(role, history, simulation_score, assessment_data)

    conn = get_db()
    conn.execute(
        "INSERT INTO results (user_id,role,overall,rating,job_ready,summary) VALUES (?,?,?,?,?,?)",
        (session["user_id"], role, report["overall"], report["rating"], report["job_ready"], report["summary"])
    )
    conn.commit()
    conn.close()

    return render_template("result.html", role=role, report=report, history=history,
                           simulation_score=simulation_score, sim_feedback=sim_feedback,
                           simulation_task=simulation_task, assessment_data=assessment_data)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    create_tables()
    app.run(debug=True)