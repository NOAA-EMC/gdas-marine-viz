#!/usr/bin/env python3
"""
Generate an HTML index file to organize observation statistics figures.

This script scans a directory for PNG files and creates an interactive HTML index
that organizes the figures by observation type, ocean basin, and quality control status.
It supports both regular figures and depth-stratified figures with depth layer information.

Usage:
    python generate_index.py /path/to/figures/directory
    python generate_index.py /path/to/figures/directory --title "Custom Title"
    python generate_index.py /path/to/figures/directory --experiments "exp1, exp2"
"""

import os
import sys
import argparse
from collections import defaultdict


def parse_filename(filename):
    """
    Parse a PNG filename to extract observation type, QC status, ocean basin, and depth layer.

    Expected formats:
    - instrument_ombg_qc_Ocean.png
    - instrument_ombg_noqc_Ocean.png
    - instrument_ombg_qc_Ocean_depth.png (with depth layer)
    - instrument_ombg_noqc_Ocean_depth.png (with depth layer)

    Returns:
        tuple: (instrument, qc_status, ocean, depth_layer)
    """
    # Remove .png extension
    basename = filename.replace('.png', '')

    # Define ocean basins
    oceans = ['Arctic', 'Atlantic', 'Indian', 'Pacific', 'Southern', 'Global']

    # Find ocean basin in filename
    ocean = None
    for o in oceans:
        if o in basename:
            ocean = o
            break

    if not ocean:
        return None, None, None, None

    # Split by ocean to get parts before and after
    parts = basename.split(f'_{ocean}')
    if len(parts) != 2:
        return None, None, None, None

    before_ocean = parts[0]
    after_ocean = parts[1]

    # Determine QC status
    if before_ocean.endswith('_ombg_qc'):
        qc_status = 'qc'
        instrument = before_ocean.replace('_ombg_qc', '')
    elif before_ocean.endswith('_ombg_noqc'):
        qc_status = 'noqc'
        instrument = before_ocean.replace('_ombg_noqc', '')
    else:
        return None, None, None, None

    # Check for depth layer in the part after ocean
    depth_layer = None
    if after_ocean:
        # Remove leading underscore if present
        after_ocean = after_ocean.lstrip('_')
        if after_ocean:
            depth_layer = after_ocean

    return instrument, qc_status, ocean, depth_layer


def organize_files(directory):
    """
    Organize PNG files by instrument type, extracting metadata from filenames.

    Returns:
        dict: Organized structure with instruments, oceans, QC status, and depth layers
    """
    files_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str))))

    # Get all PNG files in directory
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    for filename in png_files:
        instrument, qc_status, ocean, depth_layer = parse_filename(filename)

        if instrument and qc_status and ocean:
            key = f"{instrument}_{qc_status}_{ocean}"
            if depth_layer:
                key += f"_{depth_layer}"
            files_data[instrument][ocean][qc_status][depth_layer or 'surface'] = filename

    return files_data


def get_unique_instruments(files_data):
    """Get sorted list of unique instrument types."""
    return sorted(files_data.keys())


def get_unique_oceans(files_data):
    """Get sorted list of unique ocean basins."""
    oceans = set()
    for instrument_data in files_data.values():
        oceans.update(instrument_data.keys())

    # Preferred order for oceans
    ocean_order = ['Global', 'Arctic', 'Atlantic', 'Indian', 'Pacific', 'Southern']
    return [ocean for ocean in ocean_order if ocean in oceans]


def get_depth_layers(files_data, instrument):
    """Get all depth layers for a given instrument."""
    depth_layers = set()
    for ocean_data in files_data[instrument].values():
        for qc_data in ocean_data.values():
            depth_layers.update(qc_data.keys())

    # Remove 'surface' and sort the rest
    depth_layers.discard('surface')
    return ['surface'] + sorted(depth_layers) if depth_layers else ['surface']


def has_depth_layers(files_data, instrument):
    """Check if an instrument has depth-stratified data."""
    for ocean_data in files_data[instrument].values():
        for qc_data in ocean_data.values():
            if any(depth != 'surface' for depth in qc_data.keys()):
                return True
    return False


def generate_html(files_data, title=None, experiments=None, output_file='index.html'):
    """
    Generate HTML index file for the organized figures.
    """
    instruments = get_unique_instruments(files_data)
    oceans = get_unique_oceans(files_data)

    if not instruments:
        print("No valid PNG files found to organize.")
        return

    # Set default title if not provided
    if not title:
        title = "Time Series of (Observation - Background) Statistics"

    # Set default experiments if not provided
    if not experiments:
        experiments = "Marine Observation Statistics"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 1.8em;
            border-bottom: 3px solid #3498db;
            display: inline-block;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .header p {{
            font-size: 1.2em;
            color: #34495e;
            margin: 5px 0;
        }}
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .control-group {{
            margin-bottom: 15px;
        }}
        .control-label {{
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
            color: #2c3e50;
        }}
        .button-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .button {{
            padding: 8px 16px;
            cursor: pointer;
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 0.9em;
            transition: all 0.3s ease;
        }}
        .button:hover {{
            background-color: #0056b3;
            transform: translateY(-2px);
        }}
        .button.active {{
            background-color: #0056b3;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        select {{
            padding: 8px 12px;
            font-size: 1em;
            border-radius: 5px;
            border: 1px solid #ddd;
            background-color: white;
            cursor: pointer;
            min-width: 200px;
        }}
        .depth-controls {{
            display: none;
            margin-top: 15px;
        }}
        .gallery {{
            display: none;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-wrapper {{
            text-align: center;
        }}
        .image-wrapper img {{
            width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .image-wrapper img:hover {{
            transform: scale(1.02);
            border-color: #3498db;
        }}
        .image-title {{
            margin-top: 10px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .no-data {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            padding: 40px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>{experiments}</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label class="control-label">Observation Type:</label>
            <select id="typeSelect" onchange="onTypeChange(this.value)">
                <option value="">Select observation type...</option>"""

    # Add instrument options
    for instrument in instruments:
        html_content += f"""
                <option value="{instrument}">{instrument.replace('_', ' ').title()}</option>"""

    html_content += """
            </select>
        </div>

        <div class="depth-controls" id="depthControls">
            <div class="control-group">
                <label class="control-label">Depth Layer:</label>
                <select id="depthSelect" onchange="onDepthChange(this.value)">
                </select>
            </div>
        </div>

        <div class="control-group">
            <label class="control-label">Ocean Basin:</label>
            <div class="button-group">"""

    # Add ocean buttons
    for ocean in oceans:
        html_content += f"""
                <button class="button" id="{ocean}" onclick="showImages('{ocean}')">{ocean}</button>"""

    html_content += """
            </div>
        </div>
    </div>

    <div class="gallery" id="gallery">
        <div id="imageContainer"></div>
    </div>

    <script>
        // Global variables
        var currentType = '';
        var currentOcean = '';
        var currentDepth = 'surface';

        // File data structure"""

    # Add JavaScript data structure
    html_content += "\n        var filesData = {\n"

    for instrument in instruments:
        html_content += f"            '{instrument}': {{\n"
        for ocean in oceans:
            if ocean in files_data[instrument]:
                html_content += f"                '{ocean}': {{\n"
                for qc_status in ['noqc', 'qc']:
                    if qc_status in files_data[instrument][ocean]:
                        html_content += f"                    '{qc_status}': {{\n"
                        for depth, filename in files_data[instrument][ocean][qc_status].items():
                            html_content += f"                        '{depth}': '{filename}',\n"
                        html_content += "                    },\n"
                html_content += "                },\n"
        html_content += "            },\n"

    html_content += "        };\n"

    # Add depth layers data
    html_content += "\n        var depthLayers = {\n"
    for instrument in instruments:
        depth_layers = get_depth_layers(files_data, instrument)
        html_content += f"            '{instrument}': {depth_layers},\n"
    html_content += "        };\n"

    # Add the rest of the JavaScript
    html_content += """

        function onTypeChange(type) {
            currentType = type;

            // Show/hide depth controls based on whether instrument has depth layers
            var depthControls = document.getElementById('depthControls');
            var depthSelect = document.getElementById('depthSelect');

            if (type && depthLayers[type] && depthLayers[type].length > 1) {
                // Populate depth options
                depthSelect.innerHTML = '';
                depthLayers[type].forEach(function(depth) {
                    var option = document.createElement('option');
                    option.value = depth;
                    option.textContent = depth === 'surface' ? '0-bottom (all depths)' : depth.replace('_', '-');
                    if (depth === 'surface') option.selected = true;
                    depthSelect.appendChild(option);
                });
                depthControls.style.display = 'block';
                currentDepth = 'surface';
            } else {
                depthControls.style.display = 'none';
                currentDepth = 'surface';
            }

            // Clear current selection
            currentOcean = '';
            updateActiveButton('', 'button');
            document.getElementById('gallery').style.display = 'none';
        }

        function onDepthChange(depth) {
            currentDepth = depth;
            if (currentOcean) {
                showImages(currentOcean);
            }
        }

        function showImages(ocean) {
            if (!currentType) {
                alert('Please select an observation type first.');
                return;
            }

            currentOcean = ocean;
            updateActiveButton(ocean, 'button');

            var container = document.getElementById('imageContainer');
            var basePath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);

            // Check if data exists for this combination
            if (!filesData[currentType] || !filesData[currentType][ocean]) {
                container.innerHTML = '<div class="no-data">No data available for ' + currentType + ' in ' + ocean + ' ocean.</div>';
                document.getElementById('gallery').style.display = 'block';
                return;
            }

            var oceanData = filesData[currentType][ocean];
            var images = [];

            // Get filenames for noqc and qc
            var noqcFile = oceanData.noqc && oceanData.noqc[currentDepth] ? oceanData.noqc[currentDepth] : null;
            var qcFile = oceanData.qc && oceanData.qc[currentDepth] ? oceanData.qc[currentDepth] : null;

            if (noqcFile) {
                var depthLabel = currentDepth !== 'surface' ? ' (' + currentDepth.replace('_', '-') + ')' : ' (0-bottom, all depths)';
                images.push({
                    src: basePath + noqcFile,
                    title: ocean + ' - No Quality Control' + depthLabel,
                    alt: ocean + ' No QC'
                });
            }

            if (qcFile) {
                var depthLabel = currentDepth !== 'surface' ? ' (' + currentDepth.replace('_', '-') + ')' : ' (0-bottom, all depths)';
                images.push({
                    src: basePath + qcFile,
                    title: ocean + ' - Quality Control Applied' + depthLabel,
                    alt: ocean + ' QC'
                });
            }

            if (images.length === 0) {
                var depthDescription = currentDepth === 'surface' ?
                    '0-bottom (all depths)' : currentDepth.replace('_', '-') + ' depth';
                container.innerHTML = '<div class="no-data">No images available for ' + currentType + ' in ' +
                    ocean + ' ocean at ' + depthDescription + '.</div>';
            } else {
                var imageHTML = '<div class="image-grid">';
                images.forEach(function(image) {
                    imageHTML += '<div class="image-wrapper">';
                    imageHTML += '<img src="' + image.src + '" alt="' + image.alt + '" title="' + image.title + '">';
                    imageHTML += '<div class="image-title">' + image.title + '</div>';
                    imageHTML += '</div>';
                });
                imageHTML += '</div>';
                container.innerHTML = imageHTML;
            }

            document.getElementById('gallery').style.display = 'block';
        }

        function updateActiveButton(selectedId, buttonClass) {
            var buttons = document.querySelectorAll('.' + buttonClass);
            buttons.forEach(function(button) {
                if (button.id === selectedId) {
                    button.classList.add('active');
                } else {
                    button.classList.remove('active');
                }
            });
        }
    </script>
</body>
</html>"""

    return html_content


def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML index for observation statistics figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_index.py /path/to/figures
  python generate_index.py /path/to/figures --title "My Custom Title"
  python generate_index.py gfsv17sss --experiments "GFS v17 SSS vs No-SSS"
        """
    )

    parser.add_argument('directory', help='Directory containing PNG figure files')
    parser.add_argument('--title', default=None,
                        help='Title for the HTML page (default: auto-generated)')
    parser.add_argument('--experiments', default=None,
                        help='Experiment description (default: auto-generated)')
    parser.add_argument('--output', default='index.html',
                        help='Output HTML filename (default: index.html)')

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)

    # Count PNG files
    png_files = [f for f in os.listdir(args.directory) if f.endswith('.png')]
    if not png_files:
        print(f"Error: No PNG files found in '{args.directory}'.")
        sys.exit(1)

    print(f"Found {len(png_files)} PNG files in '{args.directory}'")

    # Organize files
    print("Analyzing file structure...")
    files_data = organize_files(args.directory)

    if not files_data:
        print("Error: No valid observation statistics files found.")
        print("Expected filename format: instrument_ombg_[qc|noqc]_Ocean[_depth].png")
        sys.exit(1)

    # Print summary
    instruments = get_unique_instruments(files_data)
    oceans = get_unique_oceans(files_data)
    print(f"Found {len(instruments)} instrument types: {', '.join(instruments)}")
    print(f"Found {len(oceans)} ocean basins: {', '.join(oceans)}")

    # Check for depth layers
    depth_instruments = [inst for inst in instruments if has_depth_layers(files_data, inst)]
    if depth_instruments:
        print(f"Instruments with depth layers: {', '.join(depth_instruments)}")

    # Generate HTML
    print("Generating HTML index...")
    html_content = generate_html(files_data, args.title, args.experiments, args.output)

    # Write output file
    output_path = os.path.join(args.directory, args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML index generated: {output_path}")
    print(f"Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
