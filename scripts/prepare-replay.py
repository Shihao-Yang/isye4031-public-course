#!/usr/bin/env python3
"""Build a local course-site preview with only selected classroom assets."""
from pathlib import Path
import argparse
import re
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--release', action='store_true', help='Stage the owner-approved replay in docs for publication')
args = parser.parse_args()
site = Path(__file__).resolve().parents[1]
player = site.parents[1] / 'llm-chat-replay'
subprocess.run(['npm', 'run', 'build', '--', '--base=./'], cwd=player, check=True, stdout=subprocess.DEVNULL)
subprocess.run(['bash', 'scripts/build.sh'], cwd=site, check=True)
preview = site / 'docs' if args.release else site / 'tmp/replay-preview/isye4031-public-course'
if not args.release:
    shutil.copytree(site / 'docs', preview, dirs_exist_ok=True)
replay = preview / 'replay'
replay.mkdir(parents=True, exist_ok=True)
shutil.copy2(player / 'dist/index.html', replay / 'index.html')
shutil.copytree(player / 'dist/assets', replay / 'assets', dirs_exist_ok=True)
transcript = player / 'public/examples/lecture12-replay.md'
text = transcript.read_text()
# Discover only assets linked by this lesson, plus its input data.
paths = set(re.findall(r'\]\(/(examples/lecture12/[^)]+)\)', text))
paths.add('examples/lecture12/sales_territory.csv')
(replay / 'examples').mkdir(exist_ok=True)
shutil.copy2(transcript, replay / 'examples/lecture12-replay.md')
for path in sorted(paths):
    target = replay / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(player / 'public' / path, target)
shutil.copy2(player / 'LICENSE', replay / 'LICENSE')
print(f'Local preview: {preview}')
print('Release payload staged.' if args.release else 'Local preview staged. No Git operations performed.')
