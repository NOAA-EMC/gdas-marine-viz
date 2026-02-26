#!/usr/bin/env python3
"""
Generate rasters for all satellite datasets
"""
import os
import json
import yaml
import glob
import matplotlib.pyplot as plt
from generate_generic_raster import generate_observation_raster


def generate_satellite_colorbar(output_file, vmin, vmax, cmap='RdBu_r'):
    """
    Generate a horizontal colorbar PNG for a satellite raster.

    Parameters:
    -----------
    output_file : str
        Path to save the colorbar PNG
    vmin, vmax : float
        Color scale limits
    cmap : str
        Matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.subplots_adjust(bottom=0.5)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=7)
    # Format tick labels to avoid unnecessary decimals
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels([f'{vmin:g}', f'{(vmin + vmax) / 2:g}', f'{vmax:g}'])

    plt.savefig(output_file, dpi=100, bbox_inches='tight',
                facecolor='#2a2a2a', edgecolor='none')
    plt.close()
    print(f"  Colorbar saved: {output_file}")


def load_satellite_sources(yaml_file='satellite_sources.yaml'):
    """Load satellite sources configuration from YAML file"""
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Satellite sources file not found: {yaml_file}")

    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def find_nc_file(obs_dir, base_filename):
    """
    Find NetCDF file with or without date suffix
    Looks for exact match first, then tries with wildcards

    Parameters:
    -----------
    obs_dir : str
        Directory containing observation files
    base_filename : str
        Base filename (e.g., 'sst_abi_g16_l3c.nc')

    Returns:
    --------
    str or None : Full path to found file, or None if not found
    """
    # Try exact match first (no date)
    exact_path = os.path.join(obs_dir, base_filename)
    if os.path.exists(exact_path):
        return exact_path

    # Try with date pattern (e.g., sst_abi_g16_l3c.*.nc or sst_abi_g16_l3c.2021070618.nc)
    base_no_ext = base_filename.rsplit('.', 1)[0]  # Remove .nc
    pattern = os.path.join(obs_dir, f"{base_no_ext}.*.nc")
    matches = glob.glob(pattern)

    if matches:
        # Return the first match (or could sort and take most recent)
        return matches[0]

    return None


def main():
    """Generate rasters for all SATELLITE sources"""
    results = {}

    # Load satellite sources from YAML file
    yaml_path = os.path.join(os.path.dirname(__file__), 'satellite_sources.yaml')
    SAT_SOURCES = load_satellite_sources(yaml_path)

    # Create output directory for metadata
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating SATELLITE rasters for all satellite sources...")
    print("=" * 60)

    for dataset_name, info in SAT_SOURCES.items():
        base_filename = info['filename']
        obs_dir = os.sys.argv[1] if len(os.sys.argv) > 1 else 'obs_profiles'
        nc_path = find_nc_file(obs_dir, base_filename)

        if nc_path is None:
            print(f"\nSkipping {dataset_name}: file not found")
            print(f"  Looked for: {base_filename}")
            continue

        # Get the actual filename that was found
        actual_filename = os.path.basename(nc_path)

        # Create output filename based on base filename (without date)
        base_no_ext = base_filename.rsplit('.', 1)[0]  # Remove .nc
        output_file = os.path.join(
            output_dir, f"satellite_raster_{base_no_ext}.png")

        print(f"\n{dataset_name} ({info['description']})")
        print("-" * 60)
        if actual_filename != base_filename:
            print(f"Using file: {actual_filename}")

        result = generate_observation_raster(
            nc_path,
            output_file,
            variable_name=info['variable_name'],
            vmin=info['vmin'],
            vmax=info['vmax'],
            resolution=0.5
        )

        if result:
            # Generate colorbar PNG alongside the raster
            cmap = info.get('cmap', 'RdBu_r')
            colorbar_file = os.path.join(output_dir,
                                         f"satellite_colorbar_{base_no_ext}.png")
            generate_satellite_colorbar(colorbar_file, info['vmin'], info['vmax'], cmap=cmap)

            # Store relative path from output directory (where HTML will be)
            relative_file = os.path.basename(output_file)
            relative_colorbar_file = os.path.basename(colorbar_file)
            # Use actual filename as key for metadata
            results[actual_filename] = {
                'name': dataset_name,
                'description': info['description'],
                'color': info['color'],
                'file': relative_file,
                'colorbar_file': relative_colorbar_file,
                'vmin': info['vmin'],
                'vmax': info['vmax'],
                'cmap': cmap,
                'bounds': result['bounds'],
                'n_obs': result['n_obs']
            }
            print(f"✓ Success: {result['n_obs']:,} observations")
        else:
            print("✗ Failed to generate raster")

    # Save results to JSON file in output directory
    metadata_file = os.path.join(output_dir, 'satellite_rasters_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Generated {len(results)} satellite rasters")
    print(f"Metadata saved to: {metadata_file}")

    # Print summary
    print("\nSummary:")
    for nc_file, data in results.items():
        print(f"  {data['name']:20s} - {data['n_obs']:>8,} obs")


if __name__ == '__main__':
    main()
