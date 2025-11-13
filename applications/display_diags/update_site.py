#!/usr/bin/env python3
"""Wrapper to regenerate images_manifest.json and the static gallery.

Usage:
  python3 scripts/update_site.py

This runs `scripts/generate_manifest.py` then `scripts/generate_static_gallery.py`.
"""
import subprocess
import sys
from pathlib import Path

import argparse

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description='Update manifest and static gallery')
parser.add_argument('root', nargs='?', help='project root to operate on (default: script parent)', default=None)
parser.add_argument('--out', help='optional output path to pass to generate_manifest.py', default=None)
args = parser.parse_args()
if args.root:
  ROOT = Path(args.root).resolve()

print('Updating manifest and static gallery...')

def run(cmd):
    print('>', ' '.join(cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print('Command failed with', res.returncode)
        sys.exit(res.returncode)

if __name__ == '__main__':
  gm = [sys.executable, str(ROOT / 'scripts' / 'generate_manifest.py')]
  gs = [sys.executable, str(ROOT / 'scripts' / 'generate_static_gallery.py')]
  # If a custom root was provided, pass it through
  if args.root:
    gm.append(str(ROOT))
    gs.append(str(ROOT))
  # If an out path was provided, forward to the manifest generator
  if args.out:
    gm.append('--out')
    gm.append(str(args.out))
  run(gm)
  run(gs)
  print('Update complete.')
