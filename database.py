"""
CampusVoice AI - Database Layer
--------------------------------
Simple SQLite wrapper. No ORM, kept intentionally readable for a student
project -- easy to explain line-by-line in a viva.
"""

import sqlite3
from datetime import datetime

DB_PATH = "campusvoice.db"

STATUS_FLOW = ["Submitted", "Acknowledged", "In Progress", "Resolved"]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            anonymous INTEGER DEFAULT 0,
            description TEXT NOT NULL,
            category TEXT,
            category_confidence REAL,
            urgency_score INTEGER,
            urgency_label TEXT,
            sentiment TEXT,
            status TEXT DEFAULT 'Submitted',
            is_duplicate INTEGER DEFAULT 0,
            duplicate_of_id INTEGER,
            self_help_tip TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER,
            status TEXT,
            changed_at TEXT,
            FOREIGN KEY (complaint_id) REFERENCES complaints (id)
        )
    """)
    conn.commit()
    conn.close()


def get_all_complaints(limit=200):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM complaints ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_complaint(complaint_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM complaints WHERE id = ?", (complaint_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_complaint(student_name, anonymous, description, analysis):
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    cur = conn.execute("""
        INSERT INTO complaints (
            student_name, anonymous, description, category, category_confidence,
            urgency_score, urgency_label, sentiment, status,
            is_duplicate, duplicate_of_id, self_help_tip, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Submitted', ?, ?, ?, ?, ?)
    """, (
        "Anonymous" if anonymous else student_name,
        1 if anonymous else 0,
        description,
        analysis["category"],
        analysis["category_confidence"],
        analysis["urgency_score"],
        analysis["urgency_label"],
        analysis["sentiment"],
        1 if analysis["is_duplicate"] else 0,
        analysis["duplicate_of_id"],
        analysis["self_help_tip"],
        now,
        now,
    ))
    complaint_id = cur.lastrowid
    conn.execute(
        "INSERT INTO status_history (complaint_id, status, changed_at) VALUES (?, ?, ?)",
        (complaint_id, "Submitted", now)
    )
    conn.commit()
    conn.close()
    return complaint_id


def update_status(complaint_id, new_status):
    if new_status not in STATUS_FLOW:
        raise ValueError(f"Invalid status: {new_status}")
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE complaints SET status = ?, updated_at = ? WHERE id = ?",
        (new_status, now, complaint_id)
    )
    conn.execute(
        "INSERT INTO status_history (complaint_id, status, changed_at) VALUES (?, ?, ?)",
        (complaint_id, new_status, now)
    )
    conn.commit()
    conn.close()


def get_dashboard_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) c FROM complaints").fetchone()["c"]

    by_category = conn.execute("""
        SELECT category, COUNT(*) c FROM complaints GROUP BY category ORDER BY c DESC
    """).fetchall()

    by_status = conn.execute("""
        SELECT status, COUNT(*) c FROM complaints GROUP BY status
    """).fetchall()

    by_urgency = conn.execute("""
        SELECT urgency_label, COUNT(*) c FROM complaints GROUP BY urgency_label
    """).fetchall()

    critical_open = conn.execute("""
        SELECT COUNT(*) c FROM complaints
        WHERE urgency_label = 'Critical' AND status != 'Resolved'
    """).fetchone()["c"]

    conn.close()
    return {
        "total": total,
        "by_category": [dict(r) for r in by_category],
        "by_status": [dict(r) for r in by_status],
        "by_urgency": [dict(r) for r in by_urgency],
        "critical_open": critical_open,
    }
