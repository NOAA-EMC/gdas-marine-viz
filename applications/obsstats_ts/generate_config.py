#!/usr/bin/env python3
"""
Template generator for marine observation statistics configurations.
Uses Jinja2 to generate customized YAML configurations from a template.
"""

import argparse
import yaml
from jinja2 import Environment, FileSystemLoader
import os


def generate_config(template_file, output_file, **template_vars):
    """Generate configuration file from Jinja2 template."""

    # Set up Jinja2 environment
    template_dir = os.path.dirname(template_file)
    template_name = os.path.basename(template_file)

    env = Environment(
        loader=FileSystemLoader(template_dir if template_dir else '.'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    # Load and render template
    template = env.get_template(template_name)
    rendered_config = template.render(**template_vars)

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(rendered_config)

    print(f"Generated configuration: {output_file}")

    # Validate YAML syntax
    try:
        with open(output_file, 'r') as f:
            yaml.safe_load(f)
        print("✓ YAML syntax validation passed")
    except yaml.YAMLError as e:
        print(f"✗ YAML syntax error: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate marine observation statistics configuration from template',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate default configuration (includes all platforms and basins)
  python generate_config.py -o full_config.yaml

  # Generate with custom output directory and title
  python generate_config.py -o custom_config.yaml \\
    --output-dir "./my_analysis" \\
    --title "Custom Marine Analysis"

  # Generate configuration for specific experiments only
  python generate_config.py -o cp4_only_config.yaml \\
    --experiments "cp4.02d,cp4.03"

  # Generate configuration for different dates
  python generate_config.py -o historical_config.yaml \\
    --date-wildcard "2024*" \\
    --title "Historical Analysis 2024"

  # Generate configuration for specific date range
  python generate_config.py -o december_config.yaml \\
    --date-wildcard "202412*" \\
    --title "December 2024 Analysis"
        '''
    )

    parser.add_argument('-t', '--template',
                        default='realtime_config_template.j2',
                        help='Template file (default: realtime_config_template.j2)')

    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output configuration file')

    parser.add_argument('--include-sst',
                        action='store_true',
                        default=True,
                        help='Include SST satellite observations (default: true)')

    parser.add_argument('--no-sst',
                        action='store_true',
                        help='Disable SST satellite observations')



    parser.add_argument('--output-dir',
                        help='Base output directory for plots (default: ./real-time)')

    parser.add_argument('--title',
                        help='Configuration title (default: auto-generated)')

    parser.add_argument('--experiments',
                        help='Comma-separated list of experiments to include (default: all)')

    parser.add_argument('--geovar-group',
                        default='ombg',
                        help='NetCDF geovar group (default: ombg)')

    parser.add_argument('--time-interval',
                        type=int,
                        default=21600,
                        help='Time interval in seconds (default: 21600)')

    parser.add_argument('--date-wildcard',
                        default='2025*',
                        help='Date pattern for data paths (default: 2025*, supports YYYYMMDD with wildcards)')

    args = parser.parse_args()

    # Prepare template variables
    template_vars = {}

    # Add optional parameters if provided
    if args.output_dir:
        template_vars['output_base_dir'] = args.output_dir

    if args.title:
        template_vars['config_title'] = args.title

    if args.geovar_group != 'ombg':
        template_vars['default_geovar_group'] = args.geovar_group

    if args.time_interval != 21600:
        template_vars['default_time_interval'] = args.time_interval

    if args.date_wildcard != '2025*':
        template_vars['date_wildcard'] = args.date_wildcard

    # Handle experiment filtering (for future enhancement)    # Handle experiment filtering (for future enhancement)
    if args.experiments:
        # This would require modifying the template to support experiment filtering
        print("Note: Experiment filtering not yet implemented in template")
        template_vars['selected_experiments'] = args.experiments.split(',')

    # Generate configuration
    success = generate_config(args.template, args.output, **template_vars)

    if success:
        print(f"\nTo use this configuration:")
        print(f"  python plot_timeseries.py {args.output}")
    else:
        print("\nConfiguration generation failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
