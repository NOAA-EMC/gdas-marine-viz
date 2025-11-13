#!/usr/bin/env python3
"""Generate a static gallery page by embedding images_manifest.json.

Run from project root:
    python3 scripts/generate_static_gallery.py

This writes `gallery_static.html` in the project root. Re-run after regenerating
`images_manifest.json` to update the static page.
"""
import json
from pathlib import Path
import argparse


def get_root_from_args():
  p = argparse.ArgumentParser(description='Generate static gallery embedding manifest')
  p.add_argument('root', nargs='?', help='project root containing images_manifest.json (default: script parent)', default=None)
  args = p.parse_args()
  if args.root:
    return Path(args.root).resolve()
  return Path(__file__).resolve().parents[1]


ROOT = get_root_from_args()
MANIFEST_PATH = ROOT / 'images_manifest.json'
OUT_PATH = ROOT / 'gallery_static.html'


def build_html(manifest):
  # keep manifest compact to reduce file size
  manifest_json = json.dumps(manifest, separators=(',', ':'))
  html = '''<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GDAS Verification - Static Gallery</title>
  <link href="default.css" rel="stylesheet" type="text/css" />
  <link href="fonts.css" rel="stylesheet" type="text/css" />
  <style>
    .thumb { display:flex; gap:8px; align-items:center; flex-wrap:wrap }
    img.gallery-image { width:240px; height:auto; border:1px solid #ccc; box-shadow:2px 2px 6px #888; }
    .controls { padding:12px }
    .note { font-size:0.9em; color:#ddd; margin-left:8px }
    .cycle-block { margin:20px 0; border-top:1px solid #444; padding-top:12px }
  </style>
</head>
<body>
  <div class="container">
    <div class="controls">
      <label> Cycle: <select id="cycleSelect"><option value="">(select)</option></select></label>
      <label> Obs: <select id="obsSelect"><option value="">(select)</option></select></label>
      <label> Variable: <select id="varSelect"><option value="">(select)</option></select></label>
      <span class="note">Generated static gallery. Re-run the generator to update.</span>
    </div>
    <div id="galleryRoot"></div>
  </div>

  <script>
    // Embedded manifest (generated)
    const MANIFEST = __MANIFEST_JSON__;

    function addCycleBlock(root, cycle) {
      const cb = document.createElement('div');
      cb.className = 'cycle-block';
      cb.innerHTML = `<h2>${cycle}</h2><div class="meta">Paths: <code>${cycle}/vrfy/histograms</code>, <code>${cycle}/vrfy/map_plots</code></div>`;
      root.appendChild(cb);
      return cb;
    }
    function addObsBlock(cycleBlock, obsName) {
      const ob = document.createElement('div');
      ob.className = 'obs-block';
      ob.innerHTML = `<h3>${obsName}</h3>`;
      cycleBlock.appendChild(ob);
      return ob;
    }
    function addVariableBlock(obsBlock, varName) {
      const vb = document.createElement('div');
      vb.className = 'variable-block';
      vb.innerHTML = `<h4>${varName}</h4><div class="thumb"></div>`;
      obsBlock.appendChild(vb);
      return vb.querySelector('.thumb');
    }

    function renderCycleFromManifest(cycleBlock, obj) {
      for (const top of Object.keys(obj).sort()) {
        const topObj = obj[top];
        const obsContainer = addObsBlock(cycleBlock, top);
        for (const obs of Object.keys(topObj).sort()) {
          const obsObj = topObj[obs];
          for (const variable of Object.keys(obsObj).sort()) {
            const pngs = obsObj[variable];
            if (!pngs || pngs.length === 0) continue;
            const thumbContainer = addVariableBlock(obsContainer, obs + (variable !== '_' ? (' / ' + variable) : ''));
            pngs.forEach(p => {
              const a = document.createElement('a');
              a.href = p;
              a.target = '_blank';
              a.innerHTML = `<img src="${p}" class="gallery-image" loading="lazy" alt="" title="${p}"/>`;
              thumbContainer.appendChild(a);
            });
          }
        }
      }
    }

    // Filtering helpers (same logic used by dynamic page)
    function clearSelect(id) { const s = document.getElementById(id); if (s) s.innerHTML = '<option value="">(select)</option>'; }
    function populateCycleSelect(manifest) {
      const cycleSel = document.getElementById('cycleSelect');
      cycleSel.innerHTML = '<option value="">(select)</option>';
      const cycles = Object.keys(manifest).sort().reverse();
      for (const c of cycles) {
        const opt = document.createElement('option'); opt.value = c; opt.textContent = c; cycleSel.appendChild(opt);
      }
      clearSelect('obsSelect'); clearSelect('varSelect');
      cycleSel.onchange = () => {
        const sel = cycleSel.value;
        if (!sel) { renderAll(manifest); return; }
        populateObsSelect(manifest, sel);
        const root = document.getElementById('galleryRoot'); root.innerHTML = ''; const cb = addCycleBlock(root, sel); renderCycleFromManifest(cb, manifest[sel]);
      };
    }
    function populateObsSelect(manifest, cycle) {
      const obsSel = document.getElementById('obsSelect'); obsSel.innerHTML = '<option value="">(select)</option>';
      const top = manifest[cycle] || {}; const obsSet = new Set();
      for (const topKey of Object.keys(top)) { const obj = top[topKey] || {}; for (const obs of Object.keys(obj || {})) obsSet.add(obs); }
      const obs = Array.from(obsSet).sort(); for (const o of obs) { const opt = document.createElement('option'); opt.value = o; opt.textContent = o; obsSel.appendChild(opt); }
      clearSelect('varSelect');
      obsSel.onchange = () => {
        const selObs = obsSel.value; if (!selObs) { const root = document.getElementById('galleryRoot'); root.innerHTML = ''; const cb = addCycleBlock(root, cycle); renderCycleFromManifest(cb, manifest[cycle]); return; }
        populateVarSelect(manifest, cycle, selObs);
        const root = document.getElementById('galleryRoot'); root.innerHTML = ''; const cb = addCycleBlock(root, cycle); renderFilteredCycle(manifest[cycle], cb, selObs, null);
      };
    }
    function populateVarSelect(manifest, cycle, obs) {
      const varSel = document.getElementById('varSelect'); varSel.innerHTML = '<option value="">(select)</option>';
      const top = manifest[cycle] || {}; const varSet = new Set();
      for (const topKey of Object.keys(top)) { const obj = top[topKey] || {}; const obsObj = obj[obs]; if (obsObj) { for (const v of Object.keys(obsObj)) varSet.add(v); } }
      const vars = Array.from(varSet).sort(); for (const v of vars) { const opt = document.createElement('option'); opt.value = v; opt.textContent = v; varSel.appendChild(opt); }
      varSel.onchange = () => { const selVar = varSel.value; const root = document.getElementById('galleryRoot'); root.innerHTML = ''; const cb = addCycleBlock(root, cycle); renderFilteredCycle(manifest[cycle], cb, obs, selVar); };
    }
    function renderFilteredCycle(cycleObj, cycleBlock, obsFilter, varFilter) {
      for (const top of Object.keys(cycleObj).sort()) {
        const topObj = cycleObj[top]; const obsContainer = addObsBlock(cycleBlock, top);
        for (const obs of Object.keys(topObj).sort()) {
          if (obsFilter && obs !== obsFilter) continue; const obsObj = topObj[obs];
          for (const variable of Object.keys(obsObj).sort()) {
            if (varFilter && variable !== varFilter) continue; const pngs = obsObj[variable]; if (!pngs || pngs.length === 0) continue;
            const thumbContainer = addVariableBlock(obsContainer, obs + (variable !== '_' ? (' / ' + variable) : ''));
            pngs.forEach(p => { const a = document.createElement('a'); a.href = p; a.target = '_blank'; a.innerHTML = `<img src="${p}" class="gallery-image" loading="lazy" alt="" title="${p}"/>`; thumbContainer.appendChild(a); });
          }
        }
      }
    }
    function renderAll(manifest) { const root = document.getElementById('galleryRoot'); root.innerHTML = ''; const cycles = Object.keys(manifest).sort().reverse(); for (const cycle of cycles) { const cb = addCycleBlock(root, cycle); renderCycleFromManifest(cb, manifest[cycle]); } }

    // Initialize
    window.addEventListener('load', () => { populateCycleSelect(MANIFEST); renderAll(MANIFEST); });
  </script>
</body>
</html>
'''
  # inject manifest JSON safely (avoid f-string brace conflicts)
  html = html.replace('__MANIFEST_JSON__', manifest_json)
  return html


def main():
    if not MANIFEST_PATH.exists():
        print(f'Error: manifest not found at {MANIFEST_PATH}')
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text())
    out_html = build_html(manifest)
    OUT_PATH.write_text(out_html, encoding='utf-8')
    print(f'Wrote {OUT_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
