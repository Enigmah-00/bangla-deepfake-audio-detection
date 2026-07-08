from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))

import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
PREVIEW_DIR = OUTPUT_DIR / "previews"
ASSETS = ROOT / "assets"

INK = colors.HexColor("#18212F")
MUTED = colors.HexColor("#64748B")
BLUE = colors.HexColor("#2563EB")
TEAL = colors.HexColor("#0F766E")
AMBER = colors.HexColor("#D97706")
LINE = colors.HexColor("#D9E2EC")
SOFT = colors.HexColor("#F6F8FB")


def styles():
    base = getSampleStyleSheet()
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 10
    base["Normal"].leading = 14
    base["Normal"].textColor = INK
    base["Title"].fontName = "Helvetica-Bold"
    base["Title"].fontSize = 22
    base["Title"].leading = 26
    base["Title"].textColor = INK
    base["Heading1"].fontName = "Helvetica-Bold"
    base["Heading1"].fontSize = 15
    base["Heading1"].leading = 18
    base["Heading1"].spaceBefore = 8
    base["Heading1"].spaceAfter = 6
    base["Heading1"].textColor = INK
    base["Heading2"].fontName = "Helvetica-Bold"
    base["Heading2"].fontSize = 12
    base["Heading2"].leading = 15
    base["Heading2"].spaceBefore = 6
    base["Heading2"].spaceAfter = 4
    base["Heading2"].textColor = BLUE
    base.add(
        ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontSize=8.4,
            leading=11,
            textColor=MUTED,
        )
    )
    base.add(
        ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontSize=8.2,
            leading=10.2,
            textColor=INK,
        )
    )
    base.add(
        ParagraphStyle(
            "TableHeader",
            parent=base["TableCell"],
            fontName="Helvetica-Bold",
            textColor=INK,
        )
    )
    base.add(
        ParagraphStyle(
            "Callout",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            backColor=colors.HexColor("#EEF6FF"),
            borderColor=colors.HexColor("#BFDBFE"),
            borderPadding=8,
            borderWidth=0.7,
            textColor=INK,
        )
    )
    return base


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(doc.leftMargin, 0.48 * inch, A4[0] - doc.rightMargin, 0.48 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.32 * inch, "KonthoProhori - SciBlitz AI Challenge 2026")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def p(text: str, style_name: str = "Normal") -> Paragraph:
    return Paragraph(text, STYLES[style_name])


def h(text: str, level: int = 1) -> Paragraph:
    return p(text, "Heading1" if level == 1 else "Heading2")


def bullet(text: str) -> Paragraph:
    return p(f"- {text}")


def hr():
    return HRFlowable(width="100%", color=LINE, thickness=0.8, spaceBefore=6, spaceAfter=8)


def table(data, widths=None, font_size=8.8):
    cell_style = ParagraphStyle(
        "DynamicTableCell",
        parent=STYLES["TableCell"],
        fontSize=font_size,
        leading=font_size + 2.2,
    )
    header_style = ParagraphStyle(
        "DynamicTableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )
    wrapped = []
    for row_index, row in enumerate(data):
        wrapped_row = []
        for cell in row:
            style = header_style if row_index == 0 else cell_style
            wrapped_row.append(cell if isinstance(cell, Paragraph) else Paragraph(str(cell), style))
        wrapped.append(wrapped_row)

    tbl = Table(wrapped, colWidths=widths, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 3),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def image(path: Path, width: float):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def build_report():
    out = OUTPUT_DIR / "KonthoProhori_Project_Report.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.68 * inch,
        title="KonthoProhori Project Report",
        author="KonthoProhori Team",
    )
    story = []

    story += [
        p("KonthoProhori", "Title"),
        p("Bangla Deepfake Audio Detection and Voice Forensics", "Heading2"),
        p("Track E - National Defence | SciBlitz AI Challenge 2026", "Small"),
        Spacer(1, 8),
        p(
            "KonthoProhori is a privacy-aware web application for screening suspicious "
            "Bangla voice clips. It combines exact notebook-compatible acoustic feature "
            "extraction, a saved StandardScaler, a trained XGBoost classifier, bilingual "
            "risk communication, and downloadable analysis reports.",
            "Callout",
        ),
        Spacer(1, 10),
        table(
            [
                ["Item", "Value"],
                ["Live demo", "TBD - update after final deployment"],
                ["Repository", "GitHub repository on branch/PR"],
                ["Main reported result", "99.57% accuracy on notebook random held-out split"],
                ["Important caveat", "Cross-speaker, cross-generator, and real-world validation remains future work"],
            ],
            widths=[1.55 * inch, 5.35 * inch],
        ),
        hr(),
        h("1. Problem Statement"),
        p(
            "AI-generated Bangla speech can imitate public officials, security personnel, "
            "journalists, executives, relatives, and ordinary citizens. Suspicious voice "
            "messages can spread quickly through social media and messaging platforms, "
            "creating risk for misinformation, scams, fake emergency announcements, "
            "reputational harm, and public-trust manipulation."
        ),
        p(
            "Bangla-speaking communities have limited access to locally focused tools for "
            "quickly screening suspicious audio. KonthoProhori addresses this gap as a "
            "decision-support system, not as standalone legal or forensic proof."
        ),
        h("2. Proposed Solution"),
        p(
            "The product allows a user to upload an audio file or record through the "
            "browser, then receive a three-way screening result: Likely Genuine, "
            "Inconclusive, or Likely Deepfake. The interface displays model probabilities, "
            "waveform, log-mel spectrogram, grouped acoustic evidence, limitations, and a "
            "downloadable JSON report."
        ),
        h("Target Users", 2),
        bullet("Journalists and fact-checkers who need fast screening before publishing."),
        bullet("Public communication teams and cybercrime units handling suspicious clips."),
        bullet("Financial institutions, call centers, moderators, and Bangla-speaking citizens."),
        PageBreak(),
    ]

    story += [
        h("3. Methodology"),
        p(
            "The backend validates input, decodes audio, converts it to mono, resamples to "
            "16 kHz, and analyzes the first five seconds. Invalid, empty, corrupt, "
            "unsupported, oversized, too-short, and near-silent files are rejected with "
            "safe user messages."
        ),
        table(
            [
                ["Step", "Implementation"],
                ["Decode", "SoundFile primary path with Librosa fallback where needed"],
                ["Standardization", "Mono audio, 16 kHz sample rate"],
                ["Window", "First 5.0 seconds, matching the notebook"],
                ["Validation", "Duration, finite values, size, extension, RMS, peak level"],
                ["Privacy", "No raw audio in report and no intentional retention"],
            ],
            widths=[1.7 * inch, 5.1 * inch],
        ),
        h("Feature Engineering", 2),
        table(
            [
                ["Index range", "Feature family", "Dimensions"],
                ["0:40", "MFCC mean", "40"],
                ["40:80", "MFCC standard deviation", "40"],
                ["80:92", "Chroma mean", "12"],
                ["92:220", "Mel-spectrogram mean", "128"],
                ["220:227", "Spectral-contrast mean", "7"],
                ["Total", "", "227"],
            ],
            widths=[1.4 * inch, 4.2 * inch, 1.2 * inch],
        ),
        h("4. AI/ML Approach"),
        p(
            "The classifier is the saved XGBoost binary model from the existing notebook. "
            "The application does not retrain or modify the model assets. The saved "
            "StandardScaler is applied before inference. The model returns probabilities "
            "for real and fake classes, which are communicated as model probabilities."
        ),
        table(
            [
                ["Fake probability", "Product decision"],
                [">= 0.70", "Likely Deepfake"],
                ["<= 0.30", "Likely Genuine"],
                ["Between 0.30 and 0.70", "Inconclusive"],
            ],
            widths=[2.2 * inch, 4.6 * inch],
        ),
        p(
            "These are heuristic product-level decision bands for uncertainty-aware "
            "communication. They are not newly trained thresholds.",
            "Small",
        ),
        PageBreak(),
    ]

    story += [
        h("5. Architecture And Implementation"),
        p(
            "The system uses a FastAPI backend and a Next.js frontend. The frontend handles "
            "audio selection, microphone recording, loading states, results, charts, and "
            "report download. The backend owns all ML execution and returns structured JSON."
        ),
        image(ASSETS / "architecture.png", 6.6 * inch),
        Spacer(1, 8),
        table(
            [
                ["Module", "Role"],
                ["frontend/", "User workflow and browser microphone capture"],
                ["backend/main.py", "HTTP API, CORS, safe error conversion"],
                ["src/audio_io.py", "Validation, decode, mono conversion, resampling"],
                ["src/features.py", "Exact 227-feature extraction"],
                ["src/predictor.py", "Cached scaler/model loading and prediction"],
                ["src/visualization.py", "Waveform, spectrogram, probability, evidence charts"],
                ["src/reporting.py", "Downloadable JSON result"],
            ],
            widths=[1.8 * inch, 5.0 * inch],
            font_size=8.5,
        ),
        h("Deployment", 2),
        p(
            "The recommended deployment is Railway for the FastAPI backend and Vercel for "
            "the Next.js frontend. The frontend uses NEXT_PUBLIC_API_BASE_URL to call the "
            "Railway backend. Railway serves /api/health and /api/analyze."
        ),
        PageBreak(),
    ]

    story += [
        h("6. Results"),
        p(
            "The existing notebook reports 99.57% accuracy on a random stratified held-out "
            "split. The confusion matrix is shown below. This metric is useful evidence for "
            "the available dataset but must not be presented as universal field accuracy."
        ),
        image(ASSETS / "confusion_matrix.png", 4.65 * inch),
        Spacer(1, 8),
        table(
            [
                ["Metric", "Value"],
                ["Evaluation type", "Random stratified hold-out"],
                ["Test size", "5,102"],
                ["Accuracy", "99.57%"],
                ["Real support", "2,550"],
                ["Fake support", "2,552"],
            ],
            widths=[2.1 * inch, 4.7 * inch],
        ),
        h("Validation Caveat", 2),
        p(
            "Broader cross-speaker, cross-generator, and real-world validation remains "
            "future work. Possible risks include speaker overlap, recording-condition "
            "overlap, generator-specific artifacts, compression, background noise, phone "
            "audio, and unseen synthesis systems."
        ),
        PageBreak(),
    ]

    story += [
        h("7. Impact, Privacy, And Ethics"),
        p(
            "KonthoProhori supports national-security and public-information integrity by "
            "helping users screen suspicious Bangla voice clips before trusting, forwarding, "
            "publishing, or acting on them."
        ),
        h("Responsible Use Controls", 2),
        bullet("The tool provides AI-assisted screening, not legal or forensic proof."),
        bullet("The app does not identify who is speaking."),
        bullet("Uploaded audio is not intentionally retained or used for retraining."),
        bullet("Reports exclude raw audio and absolute server paths."),
        bullet("The UI encourages source verification and expert review for high-risk cases."),
        h("Third-Party Attribution", 2),
        p(
            "Dataset reference: Bangla DeepFake Dataset by ahanaf101 on Kaggle. Libraries "
            "include FastAPI, Uvicorn, Librosa, SoundFile, NumPy, Scikit-learn, XGBoost, "
            "Joblib, Matplotlib, Next.js, React, and Lucide React."
        ),
        PageBreak(),
    ]

    story += [
        h("8. Limitations And Future Work"),
        bullet("Confirm dataset license from Kaggle before final submission."),
        bullet("Add consented real and synthetic examples if redistribution is permitted."),
        bullet("Evaluate on an external challenge set with multiple speakers and devices."),
        bullet("Run robustness tests for MP3 compression, noise, phone quality, and low volume."),
        bullet("Perform speaker-disjoint and cross-generator evaluation if metadata allows."),
        bullet("Add segment-level analysis for longer clips as an experimental feature."),
        h("Conclusion"),
        p(
            "KonthoProhori turns the existing Bangla deepfake audio model into a usable, "
            "privacy-conscious, and uncertainty-aware screening product. It is designed to "
            "support human verification and professional review, not replace them."
        ),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def build_model_card():
    out = OUTPUT_DIR / "KonthoProhori_Model_Data_Card.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.5 * inch,
        title="KonthoProhori Model and Data Card",
        author="KonthoProhori Team",
    )
    story = [
        p("KonthoProhori Model And Data Card", "Title"),
        p("One-page AI component summary for SciBlitz AI Challenge 2026", "Small"),
        hr(),
        table(
            [
                ["Dataset", "Details"],
                ["Name and source", "Bangla DeepFake Dataset by ahanaf101 on Kaggle"],
                ["URL", "https://www.kaggle.com/datasets/ahanaf101/bangla-deepfake-dataset"],
                ["Notebook usage", "25,506 files: 12,749 real and 12,757 fake"],
                ["License", "Must be manually confirmed from Kaggle before final submission"],
                ["Known limits", "Possible speaker, recording-domain, and generator overlap"],
            ],
            widths=[1.45 * inch, 5.55 * inch],
            font_size=8.2,
        ),
        Spacer(1, 6),
        table(
            [
                ["Model", "Details"],
                ["Type", "XGBoost binary classifier"],
                ["Input", "227 handcrafted acoustic features"],
                ["Preprocessing", "Mono, 16 kHz, first 5 seconds"],
                ["Scaler", "Saved StandardScaler from notebook"],
                ["Assets", "models/bangla_deepfake_detector.joblib; models/audio_scaler.joblib"],
                ["Output", "Real and fake model probabilities with three-way decision band"],
                ["Reported result", "99.57% accuracy on notebook random held-out split"],
            ],
            widths=[1.45 * inch, 5.55 * inch],
            font_size=8.2,
        ),
        Spacer(1, 6),
        table(
            [
                ["Intended Uses", "Out-of-Scope Uses"],
                [
                    "Screening, media triage, awareness, research demonstration, human-in-the-loop verification support.",
                    "Criminal conviction, automatic censorship, biometric identity verification, proof of speaker identity, unattended high-stakes decisions.",
                ],
            ],
            widths=[3.5 * inch, 3.5 * inch],
            font_size=8.2,
        ),
        Spacer(1, 6),
        table(
            [
                ["Ethical and Technical Limitations"],
                [
                    "False positives and false negatives are possible. Performance on unseen TTS systems, unseen speakers, compression, noise, music, overlapping speakers, and phone-quality audio is not established. The random split may overestimate field performance. Uploaded voice data is sensitive and should be handled with consent and care. The tool is AI-assisted screening, not legal or forensic proof."
                ],
            ],
            widths=[7.0 * inch],
            font_size=8.2,
        ),
        Spacer(1, 6),
        p(
            "Recommended next validation: external challenge set, speaker-disjoint split, "
            "cross-generator testing, compression/noise robustness table, and consented "
            "public examples.",
            "Small",
        ),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def render_previews(pdf_path: Path) -> list[Path]:
    preview_paths = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    stem = pdf_path.stem
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=1.4).to_pil()
        out = PREVIEW_DIR / f"{stem}_page_{index + 1}.png"
        bitmap.save(out)
        preview_paths.append(out)
    return preview_paths


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    card = build_model_card()
    previews = render_previews(report) + render_previews(card)
    print(f"Wrote {report.relative_to(ROOT)}")
    print(f"Wrote {card.relative_to(ROOT)}")
    print(f"Rendered {len(previews)} preview images under {PREVIEW_DIR.relative_to(ROOT)}")
    return 0


STYLES = styles()


if __name__ == "__main__":
    raise SystemExit(main())
