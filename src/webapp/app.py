"""Lightweight Flask web application for demo and small-scale testing.

Provides endpoints:
 - GET /           -> index page
 - POST /upload    -> upload a point cloud
 - POST /start     -> start processing a job
 - GET /status/<id>-> job status
 - GET /download/<id> -> download result (if ready)

This is intentionally minimal and not production-ready.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import threading
import uuid
import shutil
import os

UPLOAD_DIR = Path("./.webapp/uploads")
RESULTS_DIR = Path("./.webapp/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_jobs = {}  # jobid -> {status, input_path, result_dir, message}


def create_app():
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"), static_folder=str(Path(__file__).parent / "static"))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload():
        f = request.files.get("file")
        if f is None:
            return "no file", 400
        filename = secure_filename(f.filename)
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
        f.save(dest)
        return jsonify({"uploaded": True, "path": str(dest)})

    @app.route("/start", methods=["POST"])
    def start():
        data = request.json or request.form
        file_path = data.get("path")
        if not file_path:
            return "missing path", 400
        inp = Path(file_path)
        if not inp.exists():
            return "file not found", 404
        jobid = uuid.uuid4().hex
        job_dir = RESULTS_DIR / jobid
        job_dir.mkdir(parents=True, exist_ok=True)
        _jobs[jobid] = {"status": "queued", "input": str(inp), "result": str(job_dir), "message": "queued"}

        def _worker(jid, ip, outdir):
            _jobs[jid]["status"] = "running"
            try:
                # lazy import and run processing pipeline with TerrainCorrector
                from src.terrain_correction import TerrainCorrector
                tc = TerrainCorrector.auto_configure(n_points=100_000, precision="cm")
                tc.process_and_export(points_file=str(ip), output_dir=str(outdir), basename="webjob")
                _jobs[jid]["status"] = "done"
                _jobs[jid]["message"] = "finished"
            except Exception as e:
                _jobs[jid]["status"] = "failed"
                _jobs[jid]["message"] = str(e)

        threading.Thread(target=_worker, args=(jobid, inp, job_dir), daemon=True).start()
        return jsonify({"jobid": jobid})

    @app.route("/status/<jobid>")
    def status(jobid):
        info = _jobs.get(jobid)
        if not info:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(info)

    @app.route("/download/<jobid>")
    def download(jobid):
        info = _jobs.get(jobid)
        if not info:
            return "unknown job", 404
        if info["status"] != "done":
            return "not ready", 400
        # return zip of results directory
        outdir = Path(info["result"])
        zip_path = outdir.with_suffix(".zip")
        if zip_path.exists():
            return send_file(str(zip_path), as_attachment=True)
        # create zip
        shutil.make_archive(str(outdir), "zip", root_dir=str(outdir))
        return send_file(str(zip_path), as_attachment=True)

    return app
