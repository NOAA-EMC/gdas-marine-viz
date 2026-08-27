#!/bin/bash
#
# Generate Interactive HTML Map for Ocean Observations
#
# Usage: ./generate_html_map.sh [yyyy] [mm] [dd] [cyc] [expname] [--skip-png]
#

set -e  # Exit on error

# Check for --skip-png flag
SKIP_PNG=false
remaining_args=()
for arg in "$@"; do
    if [ "$arg" = "--skip-png" ]; then
        SKIP_PNG=true
    else
        remaining_args+=("$arg")
    fi
done

# User configuration
# ------------------
# Default values
yyyy="${remaining_args[0]:-2024}"
mm="${remaining_args[1]:-09}"
dd="${remaining_args[2]:-28}"
cyc="${remaining_args[3]:-00}"
expname="${remaining_args[4]:-retrov17_01_stream2}"


cycle="gdas.${yyyy}-${mm}-${dd}-${cyc}z"
expbase="/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfs-dev/${expname}/COMROOT/${expname}/"
expdir="${expbase}/gdas.${yyyy}${mm}${dd}/${cyc}/"
expdir_enkf="${expbase}/enkfgdas.${yyyy}${mm}${dd}/${cyc}/"

marineviz_dir="/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfs-dev/gdas-marine-viz"
diags_dir="${expdir}/analysis/ocean/diags/"
ocn_bkg="${expdir}/model/ocean/history/gdas.t${cyc}z.inst.f006.nc"
ocn_jedi_inc="${expdir}/analysis/ocean/gdas.t${cyc}z.jedi_increment.i006.nc"
ocn_mom6_inc="${expdir}/analysis/ocean/gdas.t${cyc}z.mom6_increment.i006.nc"
ocn_bkgerr_parametric="${expdir}/bmatrix/ocean/gdas.t${cyc}z.bkgerr_parametric_stddev.nc"
ocn_recentering_error="${expdir}/bmatrix/ocean/gdas.t${cyc}z.recentering_error.nc"
ice_bkgerr_parametric="${expdir}/bmatrix/ice/gdas.t${cyc}z.bkgerr_parametric_stddev.nc"
ocn_ensmble_spread="${expdir_enkf}/ensstat/analysis/ocean/enkfgdas.t${cyc}z.bg_ensvar.nc"
ice_ensmble_spread="${expdir_enkf}/ensstat/analysis/ice/enkfgdas.t${cyc}z.bg_ensvar.nc"

ice_bkg="${expdir}/model/ice/history/gdas.t${cyc}z.inst.f006.nc"
ice_jedi_inc="${expdir}/analysis/ice/gdas.t${cyc}z.jedi_increment.i006.nc"
soca_grid="/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfsv17/gfs-base/COMROOT/gfs-base/gdas.20251011/00/analysis/ocean/gdas.t00z.jedi_gridspec.tm03.nc"
soca_grid_lowres="/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gfsv17/marineanlvar/anl_geom/soca_gridspec.nc"
soca_h_lowres="/scratch3/NCEPDEV/da/Guillaume.Vernieres/common/monitor_rt/lowres_h.nc"

# End of user configuration

# Create work directory
mkdir -p scratch-${expname}-${cycle}
cd scratch-${expname}-${cycle}

echo "Step 1/3: Generating profile plots..."
if [ "$SKIP_PNG" = true ]; then
    echo "  --skip-png: Skipping PNG generation, jumping to HTML..."
else

#obs_sources="argo glider pirata rama taotriton tesac"
obs_sources="argo glider pirata rama taotriton tesac"
for var in Salt Temp; do
    for src in $obs_sources; do
        obsfile="insitu_${var,,}_profile_${src}.nc"
        if [ ! -f "${diags_dir}/${obsfile}" ]; then
            echo "Skipping ${obsfile} as it does not exist."
            continue
        fi
        python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
            --oceanfile ${ocn_bkgerr_parametric} \
            --gridfile ${soca_grid_lowres} \
            --oceanvarname "$var" \
            --hfile ${soca_h_lowres} \
            --obsfile "${diags_dir}/${obsfile}" \
            --batch_obs_profiles \
            --model_field_type bkgerr
    done
done

# Surface observations are handled directly as map markers (like drifters),
# not as profile PNGs. See --ndbc-file and --drifter-file in generate_map.py.

# Loop over background and increments
for ocn_type in bkg jedi_inc mom6_inc; do
    if [ "$ocn_type" = "bkg" ]; then
        ocn_file="${ocn_bkg}"
        ocn_cmap="gist_ncar"
        echo "Generating sections and surface plots for background..."
    elif [ "$ocn_type" = "jedi_inc" ]; then
        ocn_file="${ocn_jedi_inc}"
        ocn_cmap="jet"
        echo "Generating sections and surface plots for JEDI increment..."
    else
        ocn_file="${ocn_mom6_inc}"
        ocn_cmap="jet"
        echo "Generating sections and surface plots for MOM6 increment..."
    fi

    # Generate sections
    for var in Salt Temp; do
        if [ "$ocn_type" = "bkg" ]; then
            # Background bounds
            if [ "$var" = "Salt" ]; then
                ocean_bounds="31,38"
            else
                ocean_bounds="-2,31"
            fi
        else
            # Increment bounds
            if [ "$var" = "Salt" ]; then
                ocean_bounds="-0.5,0.5"
            else
                ocean_bounds="-2,2"
            fi
        fi

        python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
            --oceanfile ${ocn_file} \
            --gridfile ${soca_grid} \
            --oceanvarname "$var" \
            --hfile ${ocn_bkg} \
            --batch_zonal_sections \
            --lat_start -65 \
            --lat_end 65 \
            --lat_step 5 \
            --sections_output_dir "sections_${ocn_type}" \
            --ocean_bounds="${ocean_bounds}" \
            --cmap "${ocn_cmap}"

        python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
            --oceanfile ${ocn_file} \
            --gridfile ${soca_grid} \
            --oceanvarname "$var" \
            --hfile ${ocn_bkg} \
            --batch_meridional_sections \
            --lon_start -180 \
            --lon_end 180 \
            --lon_step 5 \
            --sections_output_dir "sections_${ocn_type}" \
            --ocean_bounds="${ocean_bounds}" \
            --cmap "${ocn_cmap}"
    done

    # Generate surface plots
    if [ "$ocn_type" = "bkg" ]; then
        # Background includes SSH
        for var in Temp Salt ave_ssh; do
            if [ "$var" = "Temp" ]; then
                ocean_bounds="-2,31"
            elif [ "$var" = "Salt" ]; then
                ocean_bounds="32,40"
            else
                ocean_bounds="-2,1.5"  # SSH bounds
            fi

            python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
                --oceanfile ${ocn_file} \
                --gridfile ${soca_grid} \
                --oceanvarname "$var" \
                --hfile ${ocn_bkg} \
                --batch_surface_plots \
                --use_web_mercator \
                --surface_output_dir "./surface_plots_${ocn_type}" \
                --ocean_bounds="${ocean_bounds}" \
                --cmap "${ocn_cmap}"
        done
    else
        # Increments only have Temp and Salt
        for var in Temp Salt; do
            if [ "$var" = "Salt" ]; then
                ocean_bounds="-0.5,0.5"
            else
                ocean_bounds="-2,2"
            fi

            python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
                --oceanfile ${ocn_file} \
                --gridfile ${soca_grid} \
                --oceanvarname "$var" \
                --hfile ${ocn_bkg} \
                --batch_surface_plots \
                --use_web_mercator \
                --surface_output_dir "./surface_plots_${ocn_type}" \
                --ocean_bounds="${ocean_bounds}" \
                --cmap "${ocn_cmap}"
        done
    fi
done

# Loop over sea ice background and increments
for ice_type in bkg jedi_inc; do
    if [ "$ice_type" = "bkg" ]; then
        ice_file="${ice_bkg}"
        echo "Generating sea ice surface plots for background..."
        ice_vars="aice_h sice_h hi_h hs_h"
        ice_cmap="gist_ncar"
    else
        ice_file="${ice_jedi_inc}"
        echo "Generating sea ice surface plots for JEDI increment..."
        ice_vars="aice_h hi_div_aice_h hs_div_aice_h"
        ice_cmap="jet"
    fi

    # Generate sea ice surface plots
    for var in $ice_vars; do
        if [ "$ice_type" = "bkg" ]; then
            # Background bounds
            if [ "$var" = "aice_h" ]; then
                ocean_bounds="0,1"  # Ice concentration (0-1)
            elif [ "$var" = "sice_h" ]; then
                ocean_bounds="0,35"  # Ice salinity (psu)
            elif [ "$var" = "hi_h" ]; then
                ocean_bounds="0,5"  # Ice thickness (m)
            else
                ocean_bounds="0,2"  # Snow depth (m)
            fi
        else
            # Increment bounds
            if [ "$var" = "aice_h" ]; then
                ocean_bounds="-0.2,0.2"  # Ice concentration increment
            elif [ "$var" = "hi_h" ]; then
                ocean_bounds="-0.5,0.5"  # Ice thickness increment (m)
            else
                ocean_bounds="-0.2,0.2"  # Snow depth increment (m)
            fi
        fi

        python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
            --oceanfile ${ice_file} \
            --gridfile ${soca_grid} \
            --oceanvarname "$var" \
            --hfile ${ocn_bkg} \
            --batch_surface_plots \
            --use_web_mercator \
            --surface_output_dir "./surface_plots_ice_${ice_type}" \
            --ocean_bounds="${ocean_bounds}" \
            --cmap "${ice_cmap}"
    done
done

# Loop over ocean parametric background error
echo "Generating ocean parametric background error surface plots..."
for var in Temp Salt ave_ssh; do
    if [ "$var" = "Temp" ]; then
        ocean_bounds="0,2"
    elif [ "$var" = "Salt" ]; then
        ocean_bounds="0,0.5"
    else
        ocean_bounds="0,0.2"  # SSH stddev bounds
    fi

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_bkgerr_parametric} \
        --gridfile ${soca_grid_lowres} \
        --oceanvarname "$var" \
        --hfile ${soca_h_lowres} \
        --batch_surface_plots \
        --use_web_mercator \
        --surface_output_dir "./surface_plots_ocn_bkgerr" \
        --ocean_bounds="${ocean_bounds}"
done

# Loop over ice parametric background error
echo "Generating ice parametric background error surface plots..."
for var in aice_h hi_h hs_h; do
    if [ "$var" = "aice_h" ]; then
        ocean_bounds="0,0.2"
    elif [ "$var" = "hi_h" ]; then
        ocean_bounds="0,1"
    else
        ocean_bounds="0,0.5"
    fi

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ice_bkgerr_parametric} \
        --gridfile ${soca_grid_lowres} \
        --oceanvarname "$var" \
        --hfile ${soca_h_lowres} \
        --batch_surface_plots \
        --use_web_mercator \
        --surface_output_dir "./surface_plots_ice_bkgerr" \
        --ocean_bounds="${ocean_bounds}"
done

# Recentering error (SSH only)
echo "Generating recentering error surface plot (SSH only)..."
python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
    --oceanfile ${ocn_recentering_error} \
    --gridfile ${soca_grid_lowres} \
    --oceanvarname "ave_ssh" \
    --hfile ${soca_h_lowres} \
    --batch_surface_plots \
    --use_web_mercator \
    --surface_output_dir "./surface_plots_recentering_err" \
    --ocean_bounds="0,0.2"

# Generate ocean background error sections (Temp, Salt only)
echo "Generating ocean parametric background error sections..."
for var in Temp Salt; do
    if [ "$var" = "Temp" ]; then
        ocean_bounds="0,2"
    else
        ocean_bounds="0,0.5"
    fi

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_bkgerr_parametric} \
        --gridfile ${soca_grid_lowres} \
        --oceanvarname "$var" \
        --hfile ${soca_h_lowres} \
        --batch_zonal_sections \
        --lat_start -65 \
        --lat_end 65 \
        --lat_step 5 \
        --sections_output_dir "sections_ocn_bkgerr" \
        --ocean_bounds="${ocean_bounds}"

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_bkgerr_parametric} \
        --gridfile ${soca_grid_lowres} \
        --oceanvarname "$var" \
        --hfile ${soca_h_lowres} \
        --batch_meridional_sections \
        --lon_start -180 \
        --lon_end 180 \
        --lon_step 5 \
        --sections_output_dir "sections_ocn_bkgerr" \
        --ocean_bounds="${ocean_bounds}"
done

# Loop over ocean ensemble spread
echo "Generating ocean ensemble spread surface plots..."
for var in Temp Salt; do
    if [ "$var" = "Temp" ]; then
        ocean_bounds="0,2"
    elif [ "$var" = "Salt" ]; then
        ocean_bounds="0,0.5"
    else
        ocean_bounds="0,0.2"
    fi

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_ensmble_spread} \
        --variance \
        --gridfile ${soca_grid} \
        --oceanvarname "$var" \
        --hfile ${ocn_bkg} \
        --batch_surface_plots \
        --use_web_mercator \
        --surface_output_dir "./surface_plots_ocn_ens_spread" \
        --ocean_bounds="${ocean_bounds}"
done

# Loop over ice ensemble spread
echo "Generating ice ensemble spread surface plots..."
for var in aice_h hi_h hs_h; do
    if [ "$var" = "aice_h" ]; then
        ocean_bounds="0,0.2"
    elif [ "$var" = "hi_h" ]; then
        ocean_bounds="0,1"
    else
        ocean_bounds="0,0.5"
    fi

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ice_ensmble_spread} \
        --variance \
        --gridfile ${soca_grid} \
        --oceanvarname "$var" \
        --hfile ${ocn_bkg} \
        --batch_surface_plots \
        --use_web_mercator \
        --surface_output_dir "./surface_plots_ice_ens_spread" \
        --ocean_bounds="${ocean_bounds}"
done

# Generate ocean ensemble spread sections (Temp, Salt only)
echo "Generating ocean ensemble spread sections..."
for var in Temp Salt; do
    if [ "$var" = "Temp" ]; then
        ocean_bounds="0,2"
    else
        ocean_bounds="0,0.5"
    fi

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_ensmble_spread} \
        --variance \
        --gridfile ${soca_grid} \
        --oceanvarname "$var" \
        --hfile ${ocn_bkg} \
        --batch_zonal_sections \
        --lat_start -65 \
        --lat_end 65 \
        --lat_step 5 \
        --sections_output_dir "sections_ocn_ens_spread" \
        --ocean_bounds="${ocean_bounds}"

    python3 "${marineviz_dir}/applications/aquaslice/aquaslice.py" \
        --oceanfile ${ocn_ensmble_spread} \
        --variance \
        --gridfile ${soca_grid} \
        --oceanvarname "$var" \
        --hfile ${ocn_bkg} \
        --batch_meridional_sections \
        --lon_start -180 \
        --lon_end 180 \
        --lon_step 5 \
        --sections_output_dir "sections_ocn_ens_spread" \
        --ocean_bounds="${ocean_bounds}"
done

echo "Step 2/3: Generating satellite rasters..."
python3 "${marineviz_dir}/applications/interactive_html/generate_all_sat_rasters.py" "${diags_dir}"

echo "Copy the output of aquaslice (obs_profiles) to output/obs_profiles..."
cp -r obs_profiles ./output/obs_profiles

echo "Copy the surface plots to output directories..."
for dir in surface_plots_bkg surface_plots_jedi_inc surface_plots_mom6_inc surface_plots_ice_bkg surface_plots_ice_jedi_inc surface_plots_ocn_bkgerr surface_plots_ice_bkgerr surface_plots_recentering_err surface_plots_ocn_ens_spread surface_plots_ice_ens_spread; do
    if [ -d "$dir" ]; then
        cp -r "$dir" "./output/$dir"
        echo "  Copied $dir"
    else
        echo "  Skipping $dir: directory does not exist."
    fi
done

echo "Copy the section directories to output directories..."
for dir in sections_bkg sections_jedi_inc sections_mom6_inc sections_ocn_bkgerr sections_ocn_ens_spread; do
    if [ -d "$dir" ]; then
        cp -r "$dir" "./output/$dir"
        echo "  Copied $dir"
    else
        echo "  Skipping $dir: directory does not exist."
    fi
done

fi  # end of SKIP_PNG check

echo "Step 3/3: Generating HTML map..."
python3 "${marineviz_dir}/applications/interactive_html/generate_map.py" \
            --drifter-file "${diags_dir}/insitu_temp_surface_drifter.nc" \
            --ndbc-file "${diags_dir}/insitu_temp_surface_ndbc.nc" \
            --cycle-name "$cycle" \
            --sections-dir "sections_bkg" \
            --sections-jedi-inc-dir "sections_jedi_inc" \
            --sections-mom6-inc-dir "sections_mom6_inc" \
            --surface-plots-dir "surface_plots_bkg" \
            --surface-plots-jedi-inc-dir "surface_plots_jedi_inc" \
            --surface-plots-mom6-inc-dir "surface_plots_mom6_inc" \
            --surface-plots-ice-bkg-dir "surface_plots_ice_bkg" \
            --surface-plots-ice-jedi-inc-dir "surface_plots_ice_jedi_inc" \
            --surface-plots-ocn-bkgerr-dir "surface_plots_ocn_bkgerr" \
            --surface-plots-ice-bkgerr-dir "surface_plots_ice_bkgerr" \
            --surface-plots-recentering-err-dir "surface_plots_recentering_err" \
            --sections-ocn-bkgerr-dir "sections_ocn_bkgerr" \
            --surface-plots-ocn-ens-spread-dir "surface_plots_ocn_ens_spread" \
            --surface-plots-ice-ens-spread-dir "surface_plots_ice_ens_spread" \
            --sections-ocn-ens-spread-dir "sections_ocn_ens_spread"

echo "Step 4/4: Tar the output directory"
tar -cvf ocean_observations_map_${cycle}.tar output

echo "Done! Open output/ocean-observations-map.html in a browser."
