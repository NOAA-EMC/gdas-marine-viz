#!/usr/bin/env python3
"""
Scan gdas.* cycles and produce images_manifest.json mapping:
{
  "gdas.YYYYMMDD.HH": {
    "histograms": { "obs_name": { "variable": ["path1.png", ...] } },
    "map_plots": { ... },
    "observation_scatter_plots": { ... },
    "other": { ... }
  },
  ...
}

Paths in the manifest are relative to the webroot (script directory's parent).
"""
import json
from pathlib import Path
import sys
import argparse


def get_args():
    p = argparse.ArgumentParser(description='Generate images_manifest.json from gdas.* directories')
    p.add_argument('root', nargs='?', help='project root containing gdas.* directories (default: script parent)', default=None)
    p.add_argument('--out', help='output path for manifest file (file or directory). Defaults to <root>/images_manifest.json', default=None)
    return p.parse_args()


args = get_args()
ROOT = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
# Determine output path
if args.out:
    outpath = Path(args.out)
    if outpath.is_dir() or str(outpath).endswith('/'):
        MANIFEST_OUT = outpath / 'images_manifest.json'
    else:
        MANIFEST_OUT = outpath
else:
    MANIFEST_OUT = ROOT / 'images_manifest.json'


# helper: os_walk with stable ordering
def os_walk(base: Path):
    # generator yielding (root, dirs, files) like os.walk but with sorted lists
    for root, dirnames, filenames in __import__('os').walk(base):
        dirnames.sort()
        filenames.sort()
        yield root, dirnames, filenames


def is_cycle_dir(p: Path):
    return p.is_dir() and p.name.startswith('gdas.')


def rel(p: Path):
    return p.relative_to(ROOT).as_posix()


def build_manifest():
    manifest = {}
    for cycle_dir in sorted([p for p in ROOT.iterdir() if is_cycle_dir(p)]):
        vrfy_dir = cycle_dir / 'vrfy'
        if not vrfy_dir.exists():
            continue
        cycle_key = cycle_dir.name
        manifest[cycle_key] = {}
        # Walk vrfy tree
        for root, dirs, files in os_walk(vrfy_dir):
            rootp = Path(root)
            # determine path components after vrfy
            try:
                rel_parts = rootp.relative_to(vrfy_dir).parts
            except Exception:
                rel_parts = []
            # skip if no png files
            pngs = [f for f in files if f.lower().endswith('.png')]
            if not pngs:
                continue
            # assign into manifest[cycle][type][obs][variable]
            if len(rel_parts) == 0:
                top = 'vrfy_root'
                obs = '_'
                var = '_'
            elif len(rel_parts) == 1:
                top = rel_parts[0]
                obs = '_'
                var = '_'
            elif len(rel_parts) == 2:
                top, obs = rel_parts
                var = '_'
            else:
                top = rel_parts[0]
                obs = rel_parts[1]
                var = '/'.join(rel_parts[2:])
            # set nested dicts
            manifest[cycle_key].setdefault(top, {}).setdefault(obs, {}).setdefault(var, [])
            for f in pngs:
                fp = rootp / f
                manifest[cycle_key][top][obs][var].append(rel(fp))
    return manifest


def main():
    manifest = build_manifest()
    out = MANIFEST_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print('Wrote', out)


if __name__ == '__main__':
    main()
