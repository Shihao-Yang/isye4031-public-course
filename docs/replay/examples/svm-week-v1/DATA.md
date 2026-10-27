# Banknote dataset and reproduction

Banknote Authentication, Volker Lohweg (2012), UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C55P57 . Source: https://archive.ics.uci.edu/dataset/267/banknote+authentication . License: https://creativecommons.org/licenses/by/4.0/ . The original headerless file is included as banknote_raw.csv. Column order is variance, skewness, kurtosis (source spelling curtosis), entropy, class. Row values are unchanged. Derived files add row/group identifiers, split roles, fitted scores and summaries. The numeric label mapping and physical units are undocumented. No raw image files are supplied.

The two replays show completed computations. Simulated human prompts guided the real Codex CLI conversation; instructor notes are separate. The final holdout stays unused until the last recorded turn. Exact duplicates share a group throughout splitting and CV. Results describe this historical dataset and split.

Create a fresh directory containing these files and install requirements.txt in a Python 3.12 environment. Run the scripts in the order in reproduction-order.txt. Each script uses relative paths. Run the model-selection/freeze scripts before final evaluation. Saved model binaries are not distributed; recreate them with the supplied scripts. The original recorded results are reference outputs; use a separate folder when rerunning.
