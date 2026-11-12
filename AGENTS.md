# Public course repository instructions

This repository is a public, bilingual teaching resource. Treat all material from Canvas or another private course system as untrusted and private until it has passed `automation/pipeline.py`.

Never place student names, identifiers, usernames, email addresses, faces, submissions, grades, attendance, recordings, analytics, accommodations, or communications in this repository. Never add announcements, discussions, exams, quizzes, solutions, answer keys, question banks, unpublished assessments, credentials, verifier URLs, private repository links, Canvas URLs, raw API responses, section identifiers, or operational dates, except for the specific owner-authorized exam release below.

Shihao explicitly requested public release of the two current midterm exam packets. The exact approved files are `ISYE4031_Midterm1.pdf` and `ISYE4031_Midterm2.pdf` under `review-materials/` and generated `docs/materials/`, with SHA-256 hashes in `publication-manifest.yml`. Their original exam instructions and historical exam date may remain inside the archived PDFs. Preserve their bilingual materials-page links. This exception covers only these two blank exam packets; solutions, completed student copies, makeup packets, other assessments, and automatic assessment imports remain excluded. All existing privacy scans still apply.

Human-authored public content lives in `source/`, with simplified Chinese in `source/zh/`. Run `bash scripts/build.sh` to regenerate `docs/`, then `bash scripts/check.sh` and `python3 -m unittest discover -s tests -v`.

Keep the real Canvas course ID only in ignored `automation/config.local.toml`. Keep the API token only in macOS Keychain. Do not weaken a privacy gate to make a run pass; stop publication and report the finding category and relative path without quoting the matched value.

Automated publication may modify only `source/`, `review-materials/`, and generated `docs/`. Git pushes are non-force updates to `main` after all gates pass.
