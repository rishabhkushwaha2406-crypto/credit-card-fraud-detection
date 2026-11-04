"""Build the static site by inlining the exported data into the template.

Run after `python -m src.train` (or `python -m src.export`):  python web/build.py
Writes web/index.html (deployable on Vercel, Netlify or GitHub Pages).
"""
from pathlib import Path

WEB = Path(__file__).resolve().parent
URL = "https://github.com/rishabhkushwaha2406-crypto/credit-card-fraud-detection"
DESC = ("Credit card fraud detection with Python, Pandas, Scikit-learn and XGBoost: nine models compared "
        "across SMOTE, class weights and undersampling, with a live threshold tuner and transaction scorer.")
FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
           "%3Crect x='2' y='5' width='20' height='14' rx='2.5' fill='%23070d10'/%3E"
           "%3Crect x='2' y='8.5' width='20' height='2.6' fill='%23e2bd6e'/%3E"
           "%3Ccircle cx='17' cy='15' r='2.2' fill='%23e5484d'/%3E%3C/svg%3E")

body = (WEB / "template.html").read_text(encoding="utf-8")
data = (WEB / "data" / "dashboard.json").read_text(encoding="utf-8")
assert "/*DATA*/null" in body, "template placeholder missing"
body = body.replace("/*DATA*/null", data.replace("</", "<\\/"))

head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{DESC}">
<meta name="theme-color" content="#070d10">
<meta property="og:type" content="website">
<meta property="og:title" content="Credit Card Fraud Detection">
<meta property="og:description" content="{DESC}">
<meta name="twitter:card" content="summary">
<link rel="icon" href="{FAVICON}">
<style>:root{{color-scheme:dark;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
</head>
<body>
"""
(WEB / "index.html").write_text(head + body + "\n</body>\n</html>\n", encoding="utf-8")
(WEB / "artifact.html").write_text(body, encoding="utf-8")  # body-only copy for Claude artifact hosting
print(f"Built web/index.html ({len(head + body) // 1024} KB)")
