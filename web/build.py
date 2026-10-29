"""Build web/index.html by inlining the training outputs into the template.

Run after `python -m src.train`:  python web/build.py
"""
from pathlib import Path

WEB = Path(__file__).resolve().parent
html = (WEB / "template.html").read_text()
html = html.replace("/*RESULTS*/null", (WEB / "data" / "results.json").read_text())
html = html.replace("/*MODEL*/null", (WEB / "data" / "lr_model.json").read_text())
page = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "</head>\n<body>\n" + html + "\n</body>\n</html>\n"
)
(WEB / "index.html").write_text(page)
(WEB / "artifact.html").write_text(html)  # body-only version for Claude artifact hosting
print("Built web/index.html")
