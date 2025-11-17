import os
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def build_pdf(output_path, symbol, analysis, charts):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, f"Earnings Report: {symbol}")
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 96, "Executive Summary")
    y = height - 120
    for bullet in analysis["summary"].get("bullets", [])[:10]:
        c.drawString(80, y, f"- {bullet[:110]}")
        y -= 16
        if y < 100:
            c.showPage()
            y = height - 72
    c.showPage()
    for key, data_url in charts.items():
        if not data_url:
            continue
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 72, key)
        png_bytes = BytesIO()
        png_bytes.write(b64_to_bytes(data_url))
        png_bytes.seek(0)
        img = ImageReader(png_bytes)
        c.drawImage(img, 72, height/2 - 100, width=width-144, preserveAspectRatio=True, mask='auto')
        c.showPage()
    c.save()
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

def b64_to_bytes(data_url):
    import base64
    if "," in data_url:
        data_url = data_url.split(",",1)[1]
    return base64.b64decode(data_url)
