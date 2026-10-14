# ISYE 4031 public course draft

This repository builds the bilingual public-facing version of Shihao Yang's ISYE 4031 Regression and Forecasting course.

## Purpose

The site explains an AI-agent-first approach to teaching statistics. It presents the course philosophy, learning sequence, assessment model, public course materials, and a self-contained sample lesson without copying private course operations. English is published at the site root and simplified Chinese under `/zh/`.

## Privacy boundary

This prototype must not contain:

- student names, records, submissions, grades, attendance, or recordings;
- announcements, accommodations, or other course communications;
- exams, solutions, question banks, or instructor-only assessments;
- learning-management-system exports or API responses;
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

Shihao Yang's Sites@GeorgiaTech faculty site remains the institutional discovery path. It can link to the GitHub Pages site or later host a compact English course page. Operational course systems are separate from this public site.

## 中文说明

本仓库包含 ISYE 4031《回归与预测》的中英文公开课程网站。中文版位于 `docs/zh/`，包括课程理念、教学模式、学习路径、简单线性回归示例课程和公开资料。当前讲义 PDF 为英文，并在中文资料页中明确标注。

The public site is an independently curated public teaching resource.

## Canvas-to-GitHub automation

The `automation/` workflow performs the Tuesday/Thursday refresh without placing a Canvas export or credential in Git:

1. It reads the Canvas API credential from macOS Keychain.
2. It fetches only a title-allowlisted subset of published course pages, module file/page titles, and English lecture-slide files. Assignments, quizzes, discussions, announcements, submissions, grades, analytics, and accommodations are never source inputs.
3. It reads the student roster only into memory to create an exact denylist. Roster data is never written to disk, sent to the adapting agent, or printed.
4. It strips active HTML, private links, contact information, credentials, and denylisted identifiers before Codex can read the selected teaching content.
5. Codex makes bounded edits to the bilingual Markdown under `source/`; the deterministic build regenerates `docs/`.
6. A fail-closed gate scans public text, PDF/Office metadata and extracted text, and OCR from slides/images. A separate read-only Codex pass reviews publication safety.
7. Only allowlisted public-content paths may change. A clean temporary clone is committed and pushed without force only after every gate passes.

Raw Canvas responses and temporary working copies are deleted at the end of every run. Logs contain status and finding categories, not matched values.

### Local configuration

`automation/config.example.toml` is safe to keep in Git. Copy it to ignored `automation/config.local.toml` and add the real Canvas course ID there. Never put a token in either file.

The local credential is stored as a macOS Keychain generic password under service `Codex Canvas API`; `automation/run-scheduled.sh` reads it at runtime. The current Mac also exposes `CANVAS_API_KEY` from `~/.bash_profile` using a Keychain lookup for interactive runs.

GitHub HTTPS write authentication must be available to Git before scheduling. With GitHub CLI, the one-time setup is:

```bash
gh auth login
gh auth setup-git
```

### Verify and run

```bash
source ~/.bash_profile
python3 -m unittest discover -s tests -v
python3 automation/pipeline.py --config automation/config.local.toml --preflight
python3 automation/pipeline.py --config automation/config.local.toml --dry-run --force
```

Install the local macOS schedule only after the full preflight succeeds:

```bash
python3 automation/install_schedule.py
```

The checked-in default runs at 8:30 PM local time on Tuesday and Thursday. macOS `launchd` coalesces a missed run when the Mac next wakes. Change `schedule.hour` and `schedule.minute` in the ignored local configuration if lecture timing changes.

To remove the schedule without touching the site or Keychain credential:

```bash
python3 automation/install_schedule.py --uninstall
```

Codex non-interactive mode is designed for scripted pipelines and scheduled jobs; see the [official OpenAI documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

## Lecture replay preview

Run `python3 scripts/prepare-replay.py` to build the player and stage the course site in `tmp/replay-preview/isye4031-public-course/`. Serve `tmp/replay-preview/` to test the same subdirectory used by GitHub Pages. Only the player assets, explicitly approved Lecture 12 and 13 transcripts, linked results, code, plots, input CSVs, and requirements enter the preview. Raw sessions and unrelated conversation examples are excluded. The upstream MIT license accompanies the player.

The Lecture 12 data originates from course file `5-1.txt`. On October 13, 2026, Shihao explicitly approved public release of this replay, including the dataset, after reviewing the preview and the redistribution question. This records owner authorization; it does not assert an independently verified dataset license. Run `python3 scripts/prepare-replay.py --release` and `bash scripts/check.sh` to stage and check the approved payload in `docs/`.

Shihao also approved publishing Lecture 13 after reviewing its local replay. It uses the same previously approved dataset. The original version contains seven simulated instructor prompts sent to an actual Codex CLI conversation; each turn contains real recorded Python execution. Responses are preserved with classroom asset links adjusted, and teaching annotations are labeled separately. The source textbook, legacy PDFs, private preparation notes, and raw execution logs are excluded from the release. The replay uses open-ended discussion pauses without multiple-choice questions.

The current Lecture 13 link opens version 3 with a setup reminder, brief requests for visual explanations, and nineteen recorded turns. Checkpoints 1–13 form the main route; 14–19 provide optional extensions on combining predictors, PCA, ridge, polynomial identifiability, deletion diagnostics, matching summaries and normality tests. Every response comes from a fresh Codex CLI conversation with actual Python execution. The figures appear in the response, and linked code and results support reproduction. The earlier ten-turn conversational version and original seven-turn recording remain available unchanged.

Instructor notes identify learning goals, open-ended pauses, a compact VIF calculation, and the transition to optional material. Source coverage was checked against the complete legacy topic sequence and relevant textbook sections. Old software-specific examples are replaced by worked course-data and constructed examples; a fitted factor-analysis model and full method derivations remain deferred. Private preparation contains the coverage map and a suggested 75-minute delivery guide. Public material contains no source PDFs or assessment text. Recorded responses retain their wording, with only portable asset links adjusted.

The earlier Lecture 12 hosting release addressed requests for complete worked examples throughout chapters 1 to 6, independent judgment at all eight checkpoints, a compact manual calculation in chapter 4, and checks of AI claims in chapter 7. That release changed hosting and navigation only. Assessment alignment was documented in the October 8 lesson preparation; current exam materials were not re-reviewed for that packaging change.
