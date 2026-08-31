---
title: Teaching model
description: How an AI-agent-first statistics course organizes learning, verification, and assessment.
zh_url: zh/teaching-model.html
---

<p class="eyebrow">Teaching model</p>

# From code production to statistical judgment

The redesign does not remove programming or mathematical foundations. It changes where students spend their attention. Agents can accelerate implementation, but students remain responsible for the target, assumptions, evidence, and interpretation.

## Baseline, agent, audit, defend

<div class="timeline">
<div><h3>1. Baseline</h3><p>Define the estimand or forecast target, write down the model, and establish a transparent benchmark before introducing a more complex workflow.</p></div>
<div><h3>2. Agent</h3><p>Ask an AI coding agent to plan and implement the analysis in a reproducible project. Preserve prompts, code changes, tests, and generated outputs.</p></div>
<div><h3>3. Audit</h3><p>Test the implementation against mathematical identities, simple cases, diagnostics, alternative specifications, and held-out data.</p></div>
<div><h3>4. Defend</h3><p>Explain which choices were made, what evidence supports them, what was rejected, and where the analysis can fail.</p></div>
</div>

## What the agent may do

An agent may help students:

- inspect data structure and propose an analysis plan;
- write, refactor, and test Python code;
- create diagnostic plots and model comparisons;
- explain an unfamiliar error or statistical method;
- propose checks for leakage, extrapolation, instability, and violated assumptions; and
- improve clarity after the statistical content has been verified.

## What the student must own

The student must be able to:

- state the statistical target in plain language and notation;
- justify the model and validation design;
- inspect and run every submitted component;
- identify unsupported claims or fabricated evidence;
- reproduce the analysis from a clean environment;
- disclose how AI was used; and
- answer questions about the work without relying on the agent.

## Reproducibility contract

Each substantial analysis should include:

1. a human-readable `README` with the question, data, and execution steps;
2. source code or a clearly organized notebook;
3. executable tests for important calculations or invariants;
4. a fixed environment or dependency record;
5. a concise AI usage statement; and
6. a report that separates evidence, interpretation, and limitations.

<div class="callout">
<p><strong>Why tests matter:</strong> Agent-generated code often looks reasonable even when a sign, data split, target alignment, or default option is wrong. A test makes the intended behavior explicit and repeatable.</p>
</div>

## Assessment in an AI-rich environment

No single assessment format is sufficient. The course combines:

| Evidence | What it measures |
| --- | --- |
| In-person quizzes and exams | Individual command of concepts and calculations |
| Reproducible homework | Ability to build, test, and document an analysis |
| Common forecasting challenge | Model development under a shared prediction target |
| Oral presentation and defense | Individual understanding, critique, and communication |
| AI usage statement | Transparency about workflow and verification |

The goal is not to detect whether AI was used. AI use is expected. The goal is to determine whether the student can produce and defend trustworthy statistical work.

## A course that can be evaluated

The redesign itself can be studied. Useful outcomes include conceptual performance, reproducibility failures, audit quality, oral-defense performance, time spent, accessibility, and student confidence. Comparisons should distinguish the effect of the model from the effect of prompts, scaffolding, environment, and assessment design.
