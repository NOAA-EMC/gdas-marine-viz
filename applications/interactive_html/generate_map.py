#!/usr/bin/env python3
"""
Generate an interactive HTML map for Argo profiles, surface drifters, and SST satellite data
Updated to include all 7 SST satellites with proper Web Mercator projection
"""
import os
import re
import json
import argparse
import numpy as np
try:
    import netCDF4 as nc
except ImportError:
    nc = None


def parse_filename(filename):
    """Extract variable, platform, profile ID, longitude, and latitude from filename"""
    # Pattern: insitu_salt_profile_argo_obs_profile_00000_lon-69.02_lat17.62.png
    pattern = r'insitu_(\w+)_profile_(\w+)_obs_profile_(\d+)_lon(-?\d+\.?\d*)_lat(-?\d+\.?\d*)\.png'
    match = re.match(pattern, filename)
    if match:
        variable = match.group(1)
        platform = match.group(2)
        profile_id = match.group(3)
        lon = float(match.group(4))
        lat = float(match.group(5))
        return variable, platform, profile_id, lon, lat
    return None, None, None, None, None


def get_all_profiles(profile_dir='obs_profiles'):
    """Get all profile data from the obs_profiles directory, grouped by location"""
    profiles_by_location = {}

    # Variable name mapping
    var_names = {
        'salt': 'Salinity',
        'temp': 'Temperature'
    }

    # Platform name mapping
    platform_names = {
        'argo': 'Argo'
    }

    if not os.path.exists(profile_dir):
        print(f"Error: {profile_dir} directory not found")
        return []

    for filename in sorted(os.listdir(profile_dir)):
        if filename.endswith('.png') and 'profile' in filename:
            variable, platform, profile_id, lon, lat = parse_filename(filename)
            if profile_id is not None:
                # Use lon/lat as key to group profiles at same location
                location_key = (lon, lat)

                if location_key not in profiles_by_location:
                    profiles_by_location[location_key] = {
                        'lon': lon,
                        'lat': lat,
                        'platform': platform_names.get(platform, platform),
                        'variables': {},
                        'profile_ids': set()
                    }

                # Add this variable's file
                var_display = var_names.get(variable, variable)
                if var_display not in profiles_by_location[location_key]['variables']:
                    profiles_by_location[location_key]['variables'][var_display] = []

                profiles_by_location[location_key]['variables'][var_display].append({
                    'id': profile_id,
                    'filename': filename,
                    'path': f'{profile_dir}/{filename}'
                })
                profiles_by_location[location_key]['profile_ids'].add(profile_id)

    # Convert to list format
    profiles = []
    for location_key, data in profiles_by_location.items():
        profile_entry = {
            'lon': data['lon'],
            'lat': data['lat'],
            'platform': data['platform'],
            'variables': data['variables'],
            'variable_list': list(data['variables'].keys()),
            'profile_count': len(data['profile_ids'])
        }
        profiles.append(profile_entry)

    return profiles


def get_drifter_data(nc_file='obs_profiles/insitu_temp_surface_drifter.nc'):
    """Read surface drifter data from NetCDF file"""
    if nc is None:
        print("Warning: netCDF4 not available. Install with: pip install netCDF4")
        return []

    if not os.path.exists(nc_file):
        print(f"Warning: {nc_file} not found")
        return []

    try:
        dataset = nc.Dataset(nc_file, 'r')

        # Read metadata
        lons = dataset.groups['MetaData'].variables['longitude'][:]
        lats = dataset.groups['MetaData'].variables['latitude'][:]

        # Read ombg (obs minus background)
        ombg = dataset.groups['ombg'].variables['seaSurfaceTemperature'][:]

        # Read oman (obs minus analysis)
        oman = dataset.groups['oman'].variables['seaSurfaceTemperature'][:]

        # Read QC flags
        qc_flags = dataset.groups['EffectiveQC0'].variables['seaSurfaceTemperature'][:]

        dataset.close()

        # Create drifter data list
        drifters = []
        for i in range(len(lons)):
            # Skip if any value is fill value or invalid
            if np.ma.is_masked(lons[i]) or np.ma.is_masked(lats[i]) or np.ma.is_masked(ombg[i]):
                continue
            if not np.isfinite(lons[i]) or not np.isfinite(lats[i]) or not np.isfinite(ombg[i]):
                continue

            # Convert temperature from Kelvin to Celsius (assuming data is in K)
            # OMB and OMA values remain the same (they're already temperature differences)
            drifters.append({
                'lon': float(lons[i]),
                'lat': float(lats[i]),
                'ombg': float(ombg[i]),  # Already a difference, no conversion needed
                'oman': float(oman[i]) if not np.ma.is_masked(oman[i]) and np.isfinite(oman[i]) else None,
                'qc': int(qc_flags[i]) if not np.ma.is_masked(qc_flags[i]) else None
            })

        return drifters
    except Exception as e:
        print(f"Error reading {nc_file}: {e}")
        return []


def load_sst_rasters_metadata(metadata_file='sst_rasters_metadata.json'):
    """Load SST raster metadata from JSON file"""
    if not os.path.exists(metadata_file):
        print(f"Warning: {metadata_file} not found.")
        return {}

    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_seaice_rasters_metadata(metadata_file='seaice_rasters_metadata.json'):
    """Load sea ice raster metadata from JSON file"""
    if not os.path.exists(metadata_file):
        print(f"Warning: {metadata_file} not found.")
        return {}

    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_satellite_rasters_metadata(metadata_file='output/satellite_rasters_metadata.json'):
    """Load satellite raster metadata from unified JSON file"""
    if not os.path.exists(metadata_file):
        print(f"Warning: {metadata_file} not found. Run generate_all_sat_rasters.py first.")
        return {}

    with open(metadata_file, 'r') as f:
        return json.load(f)


def get_section_images(sections_dir='sections'):
    """
    Scan for zonal and meridional section PNG files.

    Returns:
        dict: Dictionary with 'zonal' and 'meridional' keys, each containing
              a dict mapping lat/lon values to file paths
    """
    sections = {
        'zonal': {},      # latitude -> filename
        'meridional': {}  # longitude -> filename
    }

    if not os.path.exists(sections_dir):
        print(f"Warning: {sections_dir} directory not found")
        return sections

    # Pattern: zonal_section_Temp_lat+045.png or zonal_section_Salt_lat-030.png
    zonal_pattern = r'zonal_section_(\w+)_lat([+-]\d+)\.png'
    # Pattern: meridional_section_Temp_lon+120.png or meridional_section_Salt_lon-090.png
    meridional_pattern = r'meridional_section_(\w+)_lon([+-]\d+)\.png'

    for filename in os.listdir(sections_dir):
        if not filename.endswith('.png'):
            continue

        # Check for zonal sections
        match = re.match(zonal_pattern, filename)
        if match:
            varname = match.group(1)
            lat = int(match.group(2))
            # Store with variable name as part of the key
            key = (varname, lat)
            sections['zonal'][key] = os.path.join(sections_dir, filename)
            continue

        # Check for meridional sections
        match = re.match(meridional_pattern, filename)
        if match:
            varname = match.group(1)
            lon = int(match.group(2))
            # Store with variable name as part of the key
            key = (varname, lon)
            sections['meridional'][key] = os.path.join(sections_dir, filename)

    return sections


def generate_html(profiles, drifters, satellite_metadata=None, section_images=None,
                  output_file='output/ocean-observations-map.html', cycle_name=None):
    """Generate HTML map with all features"""

    # Convert satellite metadata to the format expected by HTML
    satellite_rasters_js = {}
    sst_count = 0
    seaice_count = 0
    altimetry_count = 0
    sss_count = 0

    if satellite_metadata:
        for nc_file, data in satellite_metadata.items():
            # Determine observation type based on filename patterns
            if 'icec' in nc_file or 'seaice' in nc_file.lower():
                seaice_count += 1
            elif 'rads_adt' in nc_file or 'altimetry' in nc_file.lower():
                altimetry_count += 1
            elif 'sss_' in nc_file or 'salinity' in nc_file.lower():
                sss_count += 1
            elif 'sst_' in nc_file or nc_file.startswith('sst'):
                sst_count += 1

            satellite_rasters_js[nc_file] = {
                'name': data['name'],
                'description': data['description'],
                'file': data['file'],
                'bounds': data['bounds'],
                'n_obs': data['n_obs'],
                'color': data.get('color', '#666666')
            }

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ocean Observations Interactive Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}

        #map {{
            height: 100vh;
            width: 100%;
        }}

        .info {{
            padding: 10px;
            font-size: 14px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }}

        .profile-info {{
            min-width: 300px;
        }}

        .profile-image {{
            width: 475px !important;
            max-width: none !important;
            height: auto;
            display: inline-block;
            margin: 5px;
            vertical-align: top;
        }}

        .profile-images-container {{
            display: flex;
            flex-wrap: nowrap;
            gap: 10px;
            margin: 10px 0;
        }}

        .variable-section {{
            flex: 0 0 auto;
        }}

        .variable-section h4 {{
            margin: 5px 0;
            color: #2c3e50;
            text-align: center;
        }}

        /* Enhanced styling for SST overlays */
        .sst-overlay {{
            filter: drop-shadow(0 0 2px rgba(0, 0, 0, 0.5));
        }}

        .legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            z-index: 1000;
        }}

        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }}

        .legend-item {{
            margin: 5px 0;
            display: flex;
            align-items: center;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid white;
            box-shadow: 0 0 3px rgba(0,0,0,0.3);
        }}

        .info-box {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        }}

        .info-box h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}

        .info-box p {{
            margin: 5px 0;
            color: #34495e;
        }}

        .colorbar-box {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            z-index: 1000;
            min-width: 200px;
        }}

        .colorbar-box h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            text-align: center;
            font-size: 14px;
        }}

        .colorbar-gradient {{
            width: 100%;
            height: 20px;
            background: linear-gradient(to right,
                rgb(0, 0, 255) 0%,
                rgb(127, 127, 255) 25%,
                rgb(255, 255, 255) 50%,
                rgb(255, 127, 127) 75%,
                rgb(255, 0, 0) 100%);
            border: 1px solid #ddd;
            border-radius: 3px;
            margin: 5px 0;
        }}

        .colorbar-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #555;
        }}

        .colorbar-title {{
            text-align: center;
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}

        .leaflet-tooltip-permanent {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 4px 8px;
            font-size: 11px;
            font-weight: bold;
            color: #333;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}

        .legend-box {{
            position: absolute;
            bottom: 30px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            z-index: 1000;
            min-width: 150px;
        }}

        .legend-box h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            text-align: center;
            font-size: 14px;
        }}

        .legend-box-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}

        .legend-box-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid white;
            box-shadow: 0 0 3px rgba(0,0,0,0.3);
        }}

        .legend-box-label {{
            font-size: 12px;
            color: #333;
        }}

        /* Section selector buttons */
        .section-selector {{
            position: absolute;
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid #2c3e50;
            border-radius: 5px;
            padding: 5px;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 3px;
        }}

        .section-selector-horizontal {{
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            flex-direction: row;
        }}

        .section-selector-vertical {{
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            flex-direction: column;
        }}

        .section-title {{
            font-size: 11px;
            font-weight: bold;
            color: #2c3e50;
            padding: 0 8px;
            white-space: nowrap;
        }}

        .section-buttons {{
            display: flex;
            gap: 3px;
        }}

        .section-selector-vertical .section-buttons {{
            flex-direction: column;
        }}

        .section-btn {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #3498db;
            border: 1px solid #2980b9;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .section-btn:hover {{
            background-color: #2980b9;
            transform: scale(1.3);
            box-shadow: 0 0 5px rgba(52, 152, 219, 0.5);
        }}

        .section-btn.active {{
            background-color: #e74c3c;
            border-color: #c0392b;
            box-shadow: 0 0 8px rgba(231, 76, 60, 0.6);
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <!-- Meridional section selector (horizontal, at bottom) -->
    <div class="section-selector section-selector-horizontal" id="meridional-selector">
        <span class="section-title">Meridional Sections</span>
        <div class="section-buttons" id="meridional-buttons"></div>
    </div>

    <!-- Zonal section selector (vertical, on right) -->
    <div class="section-selector section-selector-vertical" id="zonal-selector">
        <span class="section-title">Zonal Sections</span>
        <div class="section-buttons" id="zonal-buttons"></div>
    </div>

    <div class="info-box">
        <h3>Ocean Observations</h3>
        {'<p><strong>Cycle:</strong> ' + cycle_name + '</p>' if cycle_name else ''}
        <p><strong>Argo Profiles:</strong> <span id="profile-count">0</span></p>
        <p><strong>Surface Drifters:</strong> <span id="drifter-count">{len(drifters)}</span></p>
        <p><strong>SST Satellites:</strong> <span id="sst-count">{sst_count}</span></p>
        <p><strong>Altimetry:</strong> <span id="altimetry-count">{altimetry_count}</span></p>
        <p><strong>Sea Ice:</strong> <span id="seaice-count">{seaice_count}</span></p>
        <p><strong>Salinity:</strong> <span id="sss-count">{sss_count}</span></p>
    </div>

    <div class="colorbar-box">
        <h4>SST OMB Color Scale</h4>
        <div class="colorbar-gradient"></div>
        <div class="colorbar-labels">
            <span>-1°C</span>
            <span>0°C</span>
            <span>+1°C</span>
        </div>
        <div class="colorbar-title">Obs - Background</div>
    </div>

    <div class="legend-box">
        <h4>Platforms</h4>
        <div class="legend-box-item">
            <div class="legend-box-color" style="background-color: #0078d7;"></div>
            <span class="legend-box-label">Argo Profiles</span>
        </div>
        <div class="legend-box-item">
            <div class="legend-box-color" style="background-color: #666; width: 6px; height: 6px;"></div>
            <span class="legend-box-label">Surface Drifters</span>
        </div>
    </div>

    <script>
        // Profile data
        const profiles = {json.dumps(profiles, indent=8)};

        // Drifter data
        const drifters = {json.dumps(drifters, indent=8)};

        // Satellite raster data (SST and sea ice)
        const satelliteRasters = {json.dumps(satellite_rasters_js, indent=8)};

        // Section images data - convert tuple keys to nested objects
        const zonalSections = {json.dumps({str(k): v for k, v in (section_images.get('zonal', {}) if section_images else {}).items()}, indent=8)};
        const meridionalSections = {json.dumps({str(k): v for k, v in (section_images.get('meridional', {}) if section_images else {}).items()}, indent=8)};

        // Initialize the map
        const map = L.map('map', {{
            center: [0, 0],
            zoom: 2,
            maxZoom: 18,
            minZoom: 2
        }});

        // Add OpenStreetMap tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 18
        }}).addTo(map);

        // Section selection state
        let activeMeridionalSection = null;
        let activeZonalSection = null;
        let meridionalLine = null;
        let zonalLine = null;

        // Create meridional section buttons (longitude lines, -180 to 180, every 5 degrees)
        const meridionalSelector = document.getElementById('meridional-buttons');
        const meridionalLongitudes = [];
        for (let lon = -180; lon <= 180; lon += 5) {{
            meridionalLongitudes.push(lon);
            const btn = document.createElement('div');
            btn.className = 'section-btn';
            btn.title = `Meridional section at ${{lon}}°`;
            btn.dataset.longitude = lon;
            btn.onclick = () => toggleMeridionalSection(lon, btn);
            meridionalSelector.appendChild(btn);
        }}

        // Create zonal section buttons (latitude lines, -65 to 65, every 5 degrees)
        const zonalSelector = document.getElementById('zonal-buttons');
        const zonalLatitudes = [];
        for (let lat = 65; lat >= -65; lat -= 5) {{
            zonalLatitudes.push(lat);
            const btn = document.createElement('div');
            btn.className = 'section-btn';
            btn.title = `Zonal section at ${{lat}}°`;
            btn.dataset.latitude = lat;
            btn.onclick = () => toggleZonalSection(lat, btn);
            zonalSelector.appendChild(btn);
        }}

        // Function to toggle meridional section
        function toggleMeridionalSection(lon, btn) {{
            // Remove previous line if exists
            if (meridionalLine) {{
                map.removeLayer(meridionalLine);
                meridionalLine = null;
            }}

            // Deactivate previous button
            if (activeMeridionalSection !== null) {{
                document.querySelectorAll('#meridional-buttons .section-btn').forEach(b => {{
                    if (parseFloat(b.dataset.longitude) === activeMeridionalSection) {{
                        b.classList.remove('active');
                    }}
                }});
            }}

            // If clicking same button, deactivate
            if (activeMeridionalSection === lon) {{
                activeMeridionalSection = null;
                return;
            }}

            // Activate new section
            activeMeridionalSection = lon;
            btn.classList.add('active');

            // Draw meridional line (constant longitude, varies latitude)
            meridionalLine = L.polyline([
                [-90, lon],
                [90, lon]
            ], {{
                color: '#e74c3c',
                weight: 2,
                opacity: 0.7,
                dashArray: '5, 5'
            }}).addTo(map);

            // Open popup showing section info
            const popup = L.popup()
                .setLatLng([0, lon])
                .setContent(getMeridionalSectionPopupContent(lon))
                .openOn(map);

            console.log(`Meridional section selected at longitude: ${{lon}}°`);
        }}

        // Function to get meridional section popup content
        function getMeridionalSectionPopupContent(lon) {{
            // Find all sections at this longitude
            let sections = [];

            for (const key in meridionalSections) {{
                // Parse the key string like "('Temp', 120)" to extract varname and lon
                const match = key.match(/\('(\w+)',\s*([+-]?\d+)\)/);
                if (match) {{
                    const keyVarName = match[1];
                    const keyLon = parseInt(match[2]);
                    if (keyLon === lon) {{
                        sections.push({{
                            varName: keyVarName,
                            imageFile: meridionalSections[key]
                        }});
                    }}
                }}
            }}

            // Sort sections: Temp first, then Salt, then others alphabetically
            sections.sort((a, b) => {{
                if (a.varName === 'Temp') return -1;
                if (b.varName === 'Temp') return 1;
                if (a.varName === 'Salt') return -1;
                if (b.varName === 'Salt') return 1;
                return a.varName.localeCompare(b.varName);
            }});

            if (sections.length > 0) {{
                // Build HTML with sections side by side
                let imagesHtml = '<div style="display: flex; gap: 10px; flex-wrap: nowrap;">';
                sections.forEach(section => {{
                    imagesHtml += `
                        <div style="flex: 0 0 auto;">
                            <h4 style="text-align: center; margin: 5px 0;">${{section.varName}}</h4>
                            <img src="${{section.imageFile}}" alt="Meridional Section - ${{section.varName}}"
                                 style="width: 600px; height: auto; display: block;"
                                 onerror="this.parentElement.innerHTML='<p>Image not found: ${{section.imageFile}}</p>'">
                        </div>
                    `;
                }});
                imagesHtml += '</div>';

                return `
                    <div style="min-width: 300px;">
                        <h3 style="margin: 0 0 10px 0;">Meridional Section at ${{lon}}°E</h3>
                        ${{imagesHtml}}
                    </div>
                `;
            }} else {{
                return `<strong>Meridional Section</strong><br>Longitude: ${{lon}}°<br><em>No section image available</em>`;
            }}
        }}        // Function to toggle zonal section
        function toggleZonalSection(lat, btn) {{
            // Remove previous line if exists
            if (zonalLine) {{
                map.removeLayer(zonalLine);
                zonalLine = null;
            }}

            // Deactivate previous button
            if (activeZonalSection !== null) {{
                document.querySelectorAll('#zonal-buttons .section-btn').forEach(b => {{
                    if (parseFloat(b.dataset.latitude) === activeZonalSection) {{
                        b.classList.remove('active');
                    }}
                }});
            }}

            // If clicking same button, deactivate
            if (activeZonalSection === lat) {{
                activeZonalSection = null;
                return;
            }}

            // Activate new section
            activeZonalSection = lat;
            btn.classList.add('active');

            // Draw zonal line (constant latitude, varies longitude)
            zonalLine = L.polyline([
                [lat, -180],
                [lat, 180]
            ], {{
                color: '#e74c3c',
                weight: 2,
                opacity: 0.7,
                dashArray: '5, 5'
            }}).addTo(map);

            // Open popup showing section info
            const popup = L.popup()
                .setLatLng([lat, 0])
                .setContent(getZonalSectionPopupContent(lat))
                .openOn(map);

            console.log(`Zonal section selected at latitude: ${{lat}}°`);
        }}

        // Function to get zonal section popup content
        function getZonalSectionPopupContent(lat) {{
            // Find all sections at this latitude
            let sections = [];

            for (const key in zonalSections) {{
                // Parse the key string like "('Temp', 45)" to extract varname and lat
                const match = key.match(/\('(\w+)',\s*([+-]?\d+)\)/);
                if (match) {{
                    const keyVarName = match[1];
                    const keyLat = parseInt(match[2]);
                    if (keyLat === lat) {{
                        sections.push({{
                            varName: keyVarName,
                            imageFile: zonalSections[key]
                        }});
                    }}
                }}
            }}

            // Sort sections: Temp first, then Salt, then others alphabetically
            sections.sort((a, b) => {{
                if (a.varName === 'Temp') return -1;
                if (b.varName === 'Temp') return 1;
                if (a.varName === 'Salt') return -1;
                if (b.varName === 'Salt') return 1;
                return a.varName.localeCompare(b.varName);
            }});

            if (sections.length > 0) {{
                // Build HTML with sections side by side
                let imagesHtml = '<div style="display: flex; gap: 10px; flex-wrap: nowrap;">';
                sections.forEach(section => {{
                    imagesHtml += `
                        <div style="flex: 0 0 auto;">
                            <h4 style="text-align: center; margin: 5px 0;">${{section.varName}}</h4>
                            <img src="${{section.imageFile}}" alt="Zonal Section - ${{section.varName}}"
                                 style="width: 600px; height: auto; display: block;"
                                 onerror="this.parentElement.innerHTML='<p>Image not found: ${{section.imageFile}}</p>'">
                        </div>
                    `;
                }});
                imagesHtml += '</div>';

                return `
                    <div style="min-width: 300px;">
                        <h3 style="margin: 0 0 10px 0;">Zonal Section at ${{lat}}°N</h3>
                        ${{imagesHtml}}
                    </div>
                `;
            }} else {{
                return `<strong>Zonal Section</strong><br>Latitude: ${{lat}}°<br><em>No section image available</em>`;
            }}
        }}        // Platform-specific marker colors and icons
        const platformColors = {{
            'Argo': '#0078d7'
        }};

        // Create layer groups for different observation types
        const argoLayer = L.layerGroup();
        const drifterLayer = L.layerGroup();

        // Function to create colored circle marker icon
        function createMarkerIcon(color) {{
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="${{color}}" stroke="white" stroke-width="2"/></svg>`;
            return L.icon({{
                iconUrl: 'data:image/svg+xml;base64,' + btoa(svg),
                iconSize: [16, 16],
                iconAnchor: [8, 8],
                popupAnchor: [0, -8]
            }});
        }}

        // Function to create smaller drifter marker icon (no border)
        function createDrifterIcon(color) {{
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="6" height="6" viewBox="0 0 6 6"><circle cx="3" cy="3" r="2.5" fill="${{color}}" stroke="${{color}}" stroke-width="0.5"/></svg>`;
            return L.icon({{
                iconUrl: 'data:image/svg+xml;base64,' + btoa(svg),
                iconSize: [6, 6],
                iconAnchor: [3, 3],
                popupAnchor: [0, -3]
            }});
        }}

        // Function to map ombg value to color (blue-white-red scale, -1 to +1 range)
        function getOmbgColor(ombg) {{
            // Clamp value to [-1, 1] range to match satellite SST scale
            const clampedValue = Math.max(-1, Math.min(1, ombg));
            const normalized = (clampedValue + 1) / 2; // Scale to [0, 1]

            let r, g, b;
            if (normalized < 0.5) {{
                // Blue to white (0 to 0.5)
                const t = normalized * 2;
                r = Math.round(0 + (255 - 0) * t);
                g = Math.round(0 + (255 - 0) * t);
                b = 255;
            }} else {{
                // White to red (0.5 to 1)
                const t = (normalized - 0.5) * 2;
                r = 255;
                g = Math.round(255 - 255 * t);
                b = Math.round(255 - 255 * t);
            }}

            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        // Add drifter markers with color based on ombg
        drifters.forEach(drifter => {{
            const color = getOmbgColor(drifter.ombg);
            const drifterIcon = createDrifterIcon(color);

            const marker = L.marker([drifter.lat, drifter.lon], {{ icon: drifterIcon }})
                .addTo(drifterLayer);

            const omaText = drifter.oman !== null ? drifter.oman.toFixed(3) : 'N/A';
            const qcText = drifter.qc !== null ? drifter.qc : 'N/A';

            const popupContent = `
                <div class="profile-info">
                    <strong>Platform:</strong> Surface Drifter<br>
                    <strong>OMB:</strong> ${{drifter.ombg.toFixed(3)}} °C<br>
                    <strong>OMA:</strong> ${{omaText}} °C<br>
                    <strong>QC Flag:</strong> ${{qcText}}<br>
                    <strong>Longitude:</strong> ${{drifter.lon.toFixed(2)}}°<br>
                    <strong>Latitude:</strong> ${{drifter.lat.toFixed(2)}}°<br>
                </div>
            `;

            marker.bindPopup(popupContent, {{
                maxWidth: 300,
                minWidth: 200
            }});

            marker.bindTooltip(`Drifter OMB: ${{drifter.ombg.toFixed(2)}}°C<br>Lon: ${{drifter.lon.toFixed(2)}}°, Lat: ${{drifter.lat.toFixed(2)}}°`, {{
                permanent: false,
                direction: 'top',
                offset: [0, -8]
            }});
        }});

        // Add markers for each profile
        profiles.forEach(profile => {{
            const color = platformColors[profile.platform] || '#666666';
            const profileIcon = createMarkerIcon(color);

            const marker = L.marker([profile.lat, profile.lon], {{ icon: profileIcon }})
                .addTo(argoLayer);

            // Create popup content with variables displayed side by side
            let variableImages = '<div class="profile-images-container">';

            // For each variable, show the first image only (side by side)
            Object.keys(profile.variables).forEach(varName => {{
                const files = profile.variables[varName];
                if (files.length > 0) {{
                    const file = files[0]; // Use first file for each variable
                    variableImages += `
                        <div class="variable-section">
                            <h4>${{varName}}</h4>
                            <img src="${{file.path}}" alt="${{varName}} Profile" class="profile-image"
                                 onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2VlZSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTgiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+Jz">
                        </div>
                    `;
                }}
            }});

            variableImages += '</div>';

            const popupContent = `
                <div class="profile-info">
                    <strong>Platform:</strong> ${{profile.platform}}<br>
                    <strong>Variables:</strong> ${{profile.variable_list.join(', ')}}<br>
                    <strong>Longitude:</strong> ${{profile.lon.toFixed(2)}}°<br>
                    <strong>Latitude:</strong> ${{profile.lat.toFixed(2)}}°<br>
                </div>
                ${{variableImages}}
            `;

            marker.bindPopup(popupContent, {{
                maxWidth: 1000,
                minWidth: 320
            }});

            // Add tooltip with platform and variables
            const varList = profile.variable_list.join(', ');
            marker.bindTooltip(`${{profile.platform}}: ${{varList}}<br>Lon: ${{profile.lon.toFixed(2)}}°, Lat: ${{profile.lat.toFixed(2)}}°`, {{
                permanent: false,
                direction: 'top',
                offset: [0, -8],
                opacity: 0.9
            }});
        }});

        // Update profile count
        document.getElementById('profile-count').textContent = profiles.length;

        // Add layers to map (both visible by default)
        argoLayer.addTo(map);
        drifterLayer.addTo(map);

        // Create satellite raster overlays (SST and sea ice)
        const satelliteOverlays = {{}};
        Object.keys(satelliteRasters).forEach(key => {{
            const raster = satelliteRasters[key];
            satelliteOverlays[key] = L.imageOverlay(raster.file, raster.bounds, {{
                opacity: 0.8,
                interactive: false,
                className: 'sst-overlay'
            }});
        }});

        // Add layer control
        const overlayLayers = {{
            'Argo Profiles': argoLayer,
            'Surface Drifters': drifterLayer
        }};

        // Add satellite layers (SST, altimetry, sea ice, salinity)
        Object.keys(satelliteRasters).forEach(key => {{
            const raster = satelliteRasters[key];
            let prefix = 'Other';

            // Determine prefix based on filename patterns
            if (key.includes('icec') || key.includes('seaice')) {{
                prefix = 'Sea Ice';
            }} else if (key.includes('rads_adt') || key.includes('altimetry')) {{
                prefix = 'Altimetry';
            }} else if (key.includes('sss_') || key.includes('salinity')) {{
                prefix = 'Salinity';
            }} else if (key.includes('sst_') || key.startsWith('sst')) {{
                prefix = 'SST';
            }}

            overlayLayers[`${{prefix}}: ${{raster.name}} (${{raster.n_obs.toLocaleString()}} obs)`] = satelliteOverlays[key];
        }});

        L.control.layers(null, overlayLayers, {{
            collapsed: false,
            position: 'topleft'
        }}).addTo(map);
    </script>
</body>
</html>'''

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Generated {output_file}")
    print(f"  - {len(profiles)} Argo profiles")
    print(f"  - {len(drifters)} surface drifters")
    print(f"  - {sst_count} SST satellite layers")
    print(f"  - {altimetry_count} altimetry layers")
    print(f"  - {seaice_count} sea ice layers")
    print(f"  - {sss_count} salinity layers")


def main():
    """Main function to generate the map"""
    parser = argparse.ArgumentParser(
        description='Generate an interactive HTML map for ocean observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python generate_map.py

  # Specify custom directories
  python generate_map.py --output-dir my_output --profile-dir my_profiles

  # Specify custom HTML filename
  python generate_map.py --html-name my_map.html

  # Full customization
  python generate_map.py --output-dir web --profile-dir data/profiles \\
                         --drifter-file data/drifters.nc \\
                         --raster-metadata data/rasters.json \\
                         --html-name ocean_map.html
        """
    )

    parser.add_argument('--output-dir',
                        default='output',
                        help='Output directory for HTML file (default: output)')

    parser.add_argument('--html-name',
                        default='ocean-observations-map.html',
                        help='Name of the output HTML file (default: ocean-observations-map.html)')

    parser.add_argument('--profile-dir',
                        default='obs_profiles',
                        help='Directory containing profile PNG images (default: obs_profiles)')

    parser.add_argument('--drifter-file',
                        default='obs_profiles/insitu_temp_surface_drifter.nc',
                        help='Path to drifter NetCDF file (default: obs_profiles/insitu_temp_surface_drifter.nc)')

    parser.add_argument('--raster-metadata',
                        default='output/satellite_rasters_metadata.json',
                        help='Path to satellite raster metadata JSON file (default: output/satellite_rasters_metadata.json)')

    parser.add_argument('--cycle-name',
                        default=None,
                        help='Cycle name to display on the map (e.g., gdas.20210706/18)')

    parser.add_argument('--sections-dir',
                        default='sections',
                        help='Directory containing section PNG images (default: sections)')

    args = parser.parse_args()

    print("Generating interactive map...")
    print(f"  Output directory: {args.output_dir}")
    print(f"  HTML filename: {args.html_name}")
    print(f"  Profile directory: {args.profile_dir}")
    print(f"  Drifter file: {args.drifter_file}")
    print(f"  Raster metadata: {args.raster_metadata}")
    print(f"  Sections directory: {args.sections_dir}")
    if args.cycle_name:
        print(f"  Cycle name: {args.cycle_name}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get profile data
    print("\nScanning for profile images...")
    profiles = get_all_profiles(profile_dir=args.profile_dir)
    print(f"Found {len(profiles)} profile locations")

    # Get drifter data
    print("\nReading drifter data...")
    drifters = get_drifter_data(nc_file=args.drifter_file)
    print(f"Found {len(drifters)} surface drifters")

    # Load section images
    print("\nScanning for section images...")
    section_images = get_section_images(sections_dir=args.sections_dir)
    print(f"Found {len(section_images['zonal'])} zonal sections")
    print(f"Found {len(section_images['meridional'])} meridional sections")

    # Load satellite raster metadata (SST and sea ice)
    print("\nLoading satellite raster metadata...")
    satellite_metadata = load_satellite_rasters_metadata(metadata_file=args.raster_metadata)

    sst_count = sum(1 for k in satellite_metadata.keys()
                    if 'sst_' in k or k.startswith('sst'))
    altimetry_count = sum(1 for k in satellite_metadata.keys()
                          if 'rads_adt' in k or 'altimetry' in k.lower())
    seaice_count = sum(1 for k in satellite_metadata.keys()
                       if 'icec' in k or 'seaice' in k.lower())
    sss_count = sum(1 for k in satellite_metadata.keys()
                    if 'sss_' in k or 'salinity' in k.lower())

    print(f"Found {len(satellite_metadata)} satellite datasets")
    print(f"  - {sst_count} SST satellites")
    print(f"  - {altimetry_count} altimetry satellites")
    print(f"  - {seaice_count} sea ice datasets")
    print(f"  - {sss_count} salinity datasets")

    # Generate HTML
    print("\nGenerating HTML file...")
    output_path = os.path.join(args.output_dir, args.html_name)
    generate_html(profiles, drifters, satellite_metadata, section_images,
                  output_file=output_path, cycle_name=args.cycle_name)

    print(f"\nDone! Open {output_path} in a web browser.")


if __name__ == '__main__':
    main()
