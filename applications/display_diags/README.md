GDAS Verification Gallery

Overview

This small site provides two gallery options for viewing PNG outputs under `gdas.YYYYMMDD.HH` cycle directories:

- `gallery.html` — dynamic page that prefers a pre-generated `images_manifest.json` and falls back to server directory index parsing if the manifest isn't available.
- `gallery_static.html` — self-contained static page produced by `scripts/generate_static_gallery.py` that embeds the manifest JSON.

Scripts

- `scripts/generate_manifest.py [root] [--out PATH]`
  - Walks `gdas.*` cycle directories under `root` (default project root) and writes `images_manifest.json` (default) or the file specified by `--out` (file path or directory).

- `scripts/generate_static_gallery.py [root]`
  - Reads `images_manifest.json` from `root` and writes `gallery_static.html` into `root`.

- `scripts/update_site.py [root] [--out PATH]`
  - Wrapper that runs the manifest generator then the static generator; forwards `--out` to the manifest generator.

Typical usage

Regenerate the manifest and static page in the current directory:

```bash
python3 scripts/update_site.py .
```

Write the manifest to a specific file and rebuild the static gallery:

```bash
python3 scripts/update_site.py . --out ./out_manifest.json
```

Serve locally for quick verification:

```bash
python3 -m http.server 8000
# open http://localhost:8000/gallery.html or /gallery_static.html
```

Customizing appearance

- `default.css` contains the main theme styles (body background, text, buttons). Change the accent color and thumbnail size there.
- Thumbnail size is currently set to 240px; adjust the `.gallery-image` rule in `default.css` or in the static template (`scripts/generate_static_gallery.py`) if you prefer a different size.

Notes

- If your web server does not expose directory indexes, use the manifest generator and static gallery to browse images.
- The static gallery embeds the manifest JSON — re-run the generator after updating images to refresh the static page.

Contact

If you want additional features (keyboard shortcuts for the lightbox, responsive thumbnails, or a small local watch command), open an issue or request changes here.
