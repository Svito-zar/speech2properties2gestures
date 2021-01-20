from flask import Flask, abort, send_from_directory
from pathlib import Path
import re
app = Flask(__name__)
@app.route("/<experiment_id>/<file_name>.mp4")
def videos(experiment_id, file_name):
    if re.match(r"^[a-f0-9]{32}$", experiment_id) and re.match(r"^[a-z0-9\_]*$", file_name):
        path = Path("videos", experiment_id, "videos", file_name).with_suffix(".mp4")
    else:
        abort(404)
    if not path.exists():
        abort(404)
    return send_from_directory(path.parent, path.name)