#!/bin/bash
#
# Generate Interactive HTML Map for Ocean Observations
#
# Usage: ./generate_html_map.sh
#

set -e  # Exit on error

# User configuration
# ------------------
marineviz_dir=/home/gvernier/sandboxes/gdas-marine-viz-combined
diags_dir="./rt-3dvar"
#diags_dir="./newqc"
cycle="gdas.2026020300"
ocn_bkg="${diags_dir}/bkg/gdas.t00z.inst.f006.nc"
soca_grid="soca_gridspec_025.nc"
# End of user configuration

echo "Step 1/3: Generating profile plots..."
for var in Salt Temp; do
    if [ "$var" = "Salt" ]; then
        obsfile="insitu_salt_profile_argo.nc"
    else
        obsfile="insitu_temp_profile_argo.nc"
    fi
    if [ ! -f "${diags_dir}/${obsfile}" ]; then
        echo "Skipping ${obsfile} as it does not exist."
        continue
    fi
    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_bkg} \
        --gridfile ${soca_grid} \
        --oceanvarname "$var" \
        --hfile ${ocn_bkg} \
        --obsfile "${diags_dir}/${obsfile}" \
        --batch_obs_profiles --no_plot_background
done

for var in Salt Temp; do
    if [ "$var" = "Salt" ]; then
        ocean_bounds="31,38"
    else
        ocean_bounds="-2,31"
    fi

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_bkg} \
        --gridfile ${soca_grid} \
        --oceanvarname "$var" \
        --hfile ${ocn_bkg} \
        --batch_zonal_sections \
        --lat_start -65 \
        --lat_end 65 \
        --lat_step 5 \
        --ocean_bounds="${ocean_bounds}"

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_bkg} \
        --gridfile ${soca_grid} \
        --oceanvarname "$var" \
        --hfile ${ocn_bkg} \
        --batch_meridional_sections \
        --lon_start -180 \
        --lon_end 180 \
        --lon_step 5 \
        --ocean_bounds="${ocean_bounds}"
done

echo "Step 2/3: Generating satellite rasters..."
python3 "${marineviz_dir}/applications/interactive_html/generate_all_sat_rasters.py" "${diags_dir}"

echo "Copy the output of aquaslice (obs_profiles) to output/obs_profiles..."
cp -r obs_profiles ./output/obs_profiles

echo "Copy the sections directory to output/sections..."
if [ -d "sections" ]; then
    cp -r sections ./output/sections
else
    echo "Skipping copy: sections directory does not exist."
fi

echo "Step 3/3: Generating HTML map..."
python3 "${marineviz_dir}/applications/interactive_html/generate_map.py" \
            --drifter-file "${diags_dir}/insitu_temp_surface_drifter.nc" \
            --cycle-name "$cycle" \
            --sections-dir "sections"

echo "Step 4/4: Tar the output directory"
tar -cvf ocean_observations_map_${cycle}.tar output

echo "Done! Open output/ocean-observations-map.html in a browser."
