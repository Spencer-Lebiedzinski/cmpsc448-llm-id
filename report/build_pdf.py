"""Build report/report.pdf from report/report.md (Markdown -> HTML -> headless Edge/Chrome print).

  pip install markdown
  python report/build_pdf.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            shutil.which("msedge"), shutil.which("chrome"), shutil.which("chromium")]
CSS = """
@page { size: Letter; margin: 0.9in; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.4; color: #111; }
h1 { font-size: 19pt; margin-bottom: 0.1em; } h2 { font-size: 14pt; margin-top: 1.4em; border-bottom: 1px solid #999; }
h3 { font-size: 11.5pt; margin-top: 1.1em; } h2, h3 { break-after: avoid; }
table { border-collapse: collapse; margin: 0.7em 0; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3px 7px; } th { background: #eee; }
img { max-width: 100%; display: block; margin: 0.6em auto; break-inside: avoid; }
pre { background: #f4f4f4; padding: 8px; font-size: 8.5pt; white-space: pre-wrap; break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 0.92em; }
hr { border: 0; border-top: 1px solid #bbb; margin: 1.2em 0; }
"""


def main():
    body = markdown.markdown((HERE / "report.md").read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    html = HERE / "report.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Who Wrote It?</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    browser = next((b for b in BROWSERS if b and Path(b).exists()), None)
    if not browser:
        sys.exit(f"no Edge/Chrome found; open {html} and print to PDF manually")
    pdf = HERE / "report.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
