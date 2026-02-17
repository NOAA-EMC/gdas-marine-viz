#!/usr/bin/env python3
"""
Generate rasters for all satellite datasets
"""
import os
import json
import yaml
import glob
from generate_generic_raster import generate_observation_raster


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
            # Store relative path from output directory (where HTML will be)
            relative_file = os.path.basename(output_file)
            # Use actual filename as key for metadata
            results[actual_filename] = {
                'name': dataset_name,
                'description': info['description'],
                'color': info['color'],
                'file': relative_file,
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
