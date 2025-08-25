import argparse
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


def export_markdown_to_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding="utf-8")
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    x = 20 * mm
    y = height - 20 * mm
    for line in text.splitlines():
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
        c.drawString(x, y, line)
        y -= 6 * mm
    c.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outline", default="submission_outline.md")
    parser.add_argument("--output", default="submission_outline.pdf")
    args = parser.parse_args()

    md = Path(args.outline)
    pdf = Path(args.output)
    export_markdown_to_pdf(md, pdf)
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()


