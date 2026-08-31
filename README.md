# ISYE 4031 public course draft

This repository builds the bilingual public-facing version of Shihao Yang's ISYE 4031 Regression and Forecasting course.

## Purpose

The site explains an AI-agent-first approach to teaching statistics. It presents the course philosophy, learning sequence, assessment model, public course materials, and a self-contained sample lesson without mirroring private Canvas operations. English is published at the site root and simplified Chinese under `/zh/`.

## Privacy boundary

This prototype must not contain:

- student names, records, submissions, grades, attendance, or recordings;
- announcements, accommodations, or other course communications;
- exams, solutions, question banks, or instructor-only assessments;
- Canvas exports or API responses;
- the locally stored textbook or any material without confirmed redistribution permission; or
- credentials, private repository links, or hidden document metadata.

The sample PDFs in `docs/materials/` are locally staged for review. Their provenance and redistribution status must be confirmed before publication.

## Structure

- `source/`: Markdown-first public content.
- `source/zh/`: simplified Chinese public content.
- `templates/`: the shared Pandoc HTML template.
- `assets/`: site styling.
- `scripts/build.sh`: deterministic local build.
- `docs/`: generated static preview and optional GitHub Pages mirror.
- `publication-manifest.yml`: explicit public-content allowlist and release gates.
- `canvas-public-source-inventory-2026-08-31.md`: sanitized record of the read-only Canvas review.
- `hosting-decision.md`: verified Sites@GeorgiaTech publication decision and workflow.

## Build locally

```bash
bash scripts/build.sh
bash scripts/check.sh
python3 -m http.server 8000 --directory docs
```

Then open `http://127.0.0.1:8000`.

## Publication route

GitHub Pages publishes the bilingual site from the `docs/` directory. The English homepage is `docs/index.html`; the Chinese homepage is `docs/zh/index.html`. Each page links to its counterpart in the other language.

Shihao Yang's Sites@GeorgiaTech faculty site remains the institutional discovery path. It can link to the GitHub Pages site or later host a compact English course page. Canvas remains the private source for enrolled-student operations.

## 中文说明

本仓库包含 ISYE 4031《回归与预测》的中英文公开课程网站。中文版位于 `docs/zh/`，包括课程理念、教学模式、学习路径、简单线性回归示例课程和公开资料。当前讲义与作业 PDF 为英文，并在中文资料页中明确标注。

The public site is an independently curated teaching resource, not an automatic Canvas mirror.
