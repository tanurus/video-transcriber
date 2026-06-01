from __future__ import annotations

from flask import Flask, current_app, render_template


def register_routes(app: Flask) -> None:
    @app.route("/healthz")
    def healthz():  # noqa: ANN202
        return "ok", 200

    @app.route("/")
    def index():  # noqa: ANN202
        storage = current_app.config["STORAGE"]
        jobs = storage.list_jobs()[:10]
        return render_template("index.html", jobs=jobs)

    @app.route("/upload", methods=["POST"])
    def upload():  # noqa: ANN202
        return "", 501

    @app.route("/job/<job_id>")
    def job_page(job_id):  # noqa: ANN202
        return "", 501

    @app.route("/api/job/<job_id>")
    def job_api(job_id):  # noqa: ANN202
        return "", 501

    @app.route("/download/<job_id>")
    def download(job_id):  # noqa: ANN202
        return "", 501

    @app.route("/history")
    def history():  # noqa: ANN202
        return "", 501
