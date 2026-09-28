"""
CampusVoice AI - Main Flask Application
-----------------------------------------
Run with:  python3 app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import database as db
import ai_engine

app = Flask(__name__)
db.init_db()


# ---------------------------------------------------------------------------
# STUDENT-FACING ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit_complaint():
    student_name = request.form.get("student_name", "").strip()
    anonymous = request.form.get("anonymous") == "on"
    description = request.form.get("description", "").strip()

    if not description:
        return jsonify({"error": "Description cannot be empty."}), 400

    existing = db.get_all_complaints(limit=500)
    analysis = ai_engine.analyze_complaint(description, existing_complaints=existing)

    complaint_id = db.insert_complaint(
        student_name=student_name or "Student",
        anonymous=anonymous,
        description=description,
        analysis=analysis,
    )

    return jsonify({
        "success": True,
        "complaint_id": complaint_id,
        "analysis": analysis,
    })


@app.route("/track/<int:complaint_id>")
def track_complaint(complaint_id):
    complaint = db.get_complaint(complaint_id)
    if not complaint:
        return jsonify({"error": "Not found"}), 404
    return jsonify(complaint)


# ---------------------------------------------------------------------------
# ADMIN / DASHBOARD ROUTES
# ---------------------------------------------------------------------------

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/complaints")
def api_complaints():
    status_filter = request.args.get("status")
    category_filter = request.args.get("category")

    complaints = db.get_all_complaints()

    if status_filter and status_filter != "All":
        complaints = [c for c in complaints if c["status"] == status_filter]
    if category_filter and category_filter != "All":
        complaints = [c for c in complaints if c["category"] == category_filter]

    return jsonify(complaints)


@app.route("/api/complaint/<int:complaint_id>/status", methods=["POST"])
def api_update_status(complaint_id):
    new_status = request.json.get("status")
    try:
        db.update_status(complaint_id, new_status)
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_dashboard_stats())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
