Perform a final privacy and publication-safety review of this public course repository. This is read-only review.

Review only:

- visible content and links in `source/**/*.md` and `docs/**/*.html`;
- the filenames, not the contents, under `review-materials/` and `docs/materials/`.

Do not open or semantically re-evaluate PDF, Office, or image binaries. Before this audit, a deterministic gate has already filtered assignment/logistics pages and scanned binary metadata, extracted text, rendered-page OCR, credentials, private links, and the in-memory student-roster denylist. Treat that binary gate as authoritative. You may fail a binary only when its filename itself exposes a protected identifier, credential, private URL, or raw LMS ID.

Fail the HTML/Markdown audit if you find actual student names or identifiers, student-specific grades or submissions, private communications, analytics, accommodations, copied announcement or discussion content, unpublished assessments, assessment answer keys or solutions, credentials, Canvas/private URLs, live section identifiers, a current operational schedule, or raw LMS/API material. Also fail if the English and Chinese public pages expose materially different private details.

Generic pedagogical language about students, assessment methods, quizzes, exams, grading, or AI use is allowed. Public worked calculations in the sample lesson are allowed. A generic acknowledgement that the site was adapted from Canvas is allowed; Canvas URLs, IDs, API fields, and raw responses are not. The public course code ISYE 4031 and general learning sequences without dates are allowed.

Do not report or quote any sensitive value. Findings must name only a category and relative file path. Return JSON matching the supplied schema. `pass` may be true only when there are no findings.
