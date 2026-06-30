# Webapp README

This repository includes a minimal web frontend (Flask) for small-scale demos and testing. The UI allows uploading a point cloud, starting a processing job, and downloading results.

Location
--------
- `src/webapp/` contains the Flask app
  - `app.py` - Flask application
  - `templates/` - HTML templates
  - `static/` - CSS/JS assets

Run locally
-----------
1. Install dev dependencies:
   ```bash
   pip install -e .[dev]
   pip install flask
   ```
2. Start the server (from repository root):
   ```bash
   export FLASK_APP=src.webapp.app
   flask run --host=0.0.0.0 --port=5000
   ```
3. Open http://127.0.0.1:5000 in your browser.

Endpoints (minimal)
-------------------
- `GET /` - index page with upload form
- `POST /upload` - upload a point cloud file
- `POST /start` - start a processing job for an uploaded file
- `GET /status/<jobid>` - check job status (pending/running/done)
- `GET /download/<jobid>` - download results bundle

Notes
-----
- The webapp is intended as a demo and NOT for production use. For production, run behind a WSGI server (gunicorn) and enable authentication, rate limiting, and job queueing (Celery/RQ).
