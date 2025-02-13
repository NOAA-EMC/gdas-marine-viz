C----  convert surface fluxes data from GFSv16 to DATM
C----                         hyun-chul.lee@noaa.gov
      program conv_gfs2datm
      implicit none
      include 'netcdf.inc'

C C This is the name of the data file we will read. 
      character*(*) FILE_NAME, FOUT_NAME
      character STR*200,SATT*200
      parameter (FILE_NAME = "./gfs_input.nc")
      parameter (FOUT_NAME = "./gfs_output.nc")

C C We are reading 
      integer NX,NY,NT,nerr,NDIM,i,j,k,NSTR,jinv
c     parameter (NX = 1152, NY = 576, NT = 1, NDIM = 3)
      parameter (NX = 3072, NY = 1536, NT = 1, NDIM = 3)
      real*4 :: latitude(NY), longitude(NX),time_out(NT)
      integer time(NT),NF_64BIT_OFFSET,vi_indx
      real coef,Tcel,Hn,term1,term2,g0,Rsp,rho_air,UUS,VVS,UVS,Cdrag
      real PRAT_tot, PRATE_surf
      real, dimension(NX,NY,NT) ::
     &  HGT_1hybridlevel,
     &  TMP_1hybridlevel,
     &  SPFH_1hybridlevel,
     &  UGRD_1hybridlevel,
     &  VGRD_1hybridlevel,
     &  PRES_surface,
     &  HGT_surface,
     &  TMP_surface,
     &  TSOIL_0M0D1mbelowground,
     &  SOILW_0M0D1mbelowground,
     &  SOILL_0M0D1mbelowground,
     &  TSOIL_0D1M0D4mbelowground,
     &  SOILW_0D1M0D4mbelowground,
     &  SOILL_0D1M0D4mbelowground,
     &  TSOIL_0D4M1mbelowground,
     &  SOILW_0D4M1mbelowground,
     &  SOILL_0D4M1mbelowground,
     &  TSOIL_1M2mbelowground,
     &  SOILW_1M2mbelowground,
     &  SOILL_1M2mbelowground,
     &  SOILM_0M2mbelowground,
     &  CNWAT_surface,
     &  WEASD_surface,
     &  SNOD_surface,
     &  PEVPR_surface,
     &  ICETK_surface,
     &  ACOND_surface,
     &  TMP_2maboveground,
     &  SPFH_2maboveground,
     &  UGRD_10maboveground,
     &  VGRD_10maboveground,
     &  CPOFP_surface,
     &  SFCR_surface,
     &  FRICV_surface,
     &  SHTFL_surface,
     &  LHTFL_surface,
     &  SFEXC_surface,
     &  VEG_surface,
     &  GFLUX_surface,
     &  VGTYP_surface,
     &  SOTYP_surface,
     &  SLTYP_surface,
     &  WILT_surface,
     &  FLDCP_surface,
     &  SUNSD_surface,
     &  PWAT_entireatmosphere_consideredasasinglelayer_,
     &  TCDC_convectivecloudlayer,
     &  DSWRF_surface,
     &  DLWRF_surface,
     &  USWRF_surface,
     &  ULWRF_surface,
     &  HPBL_surface,
     &  LAND_surface,
     &  ICEC_surface,
     &  PRATE_surface,
     &  UFLX_surface,
     &  VFLX_surface,
     &  delz,
     &  VAR1,VAR2,vtmp

C C This will be the netCDF ID for the file and data variable.
      integer :: ncid, 
     &  vi_longitude,
     &  vi_latitude,
     &  vi_time_in,
     &  vi_HGT_1hybridlevel,
     &  vi_TMP_1hybridlevel,
     &  vi_SPFH_1hybridlevel,
     &  vi_UGRD_1hybridlevel,
     &  vi_VGRD_1hybridlevel,
     &  vi_PRES_surface,
     &  vi_HGT_surface,
     &  vi_TMP_surface,
     &  vi_TSOIL_0M0D1mbelowground,
     &  vi_SOILW_0M0D1mbelowground,
     &  vi_SOILL_0M0D1mbelowground,
     &  vi_TSOIL_0D1M0D4mbelowground,
     &  vi_SOILW_0D1M0D4mbelowground,
     &  vi_SOILL_0D1M0D4mbelowground,
     &  vi_TSOIL_0D4M1mbelowground,
     &  vi_SOILW_0D4M1mbelowground,
     &  vi_SOILL_0D4M1mbelowground,
     &  vi_TSOIL_1M2mbelowground,
     &  vi_SOILW_1M2mbelowground,
     &  vi_SOILL_1M2mbelowground,
     &  vi_SOILM_0M2mbelowground,
     &  vi_CNWAT_surface,
     &  vi_WEASD_surface,
     &  vi_SNOD_surface,
     &  vi_PEVPR_surface,
     &  vi_ICETK_surface,
     &  vi_ACOND_surface,
     &  vi_TMP_2maboveground,
     &  vi_SPFH_2maboveground,
     &  vi_UGRD_10maboveground,
     &  vi_VGRD_10maboveground,
     &  vi_CPOFP_surface,
     &  vi_SFCR_surface,
     &  vi_FRICV_surface,
     &  vi_SHTFL_surface,
     &  vi_LHTFL_surface,
     &  vi_SFEXC_surface,
     &  vi_VEG_surface,
     &  vi_GFLUX_surface,
     &  vi_VGTYP_surface,
     &  vi_SOTYP_surface,
     &  vi_SLTYP_surface,
     &  vi_WILT_surface,
     &  vi_FLDCP_surface,
     &  vi_SUNSD_surface,
     &  vi_PWAT_entireatmosphere_consideredasasinglelayer_,
     &  vi_TCDC_convectivecloudlayer,
     &  vi_DSWRF_surface,
     &  vi_DLWRF_surface,
     &  vi_USWRF_surface,
     &  vi_ULWRF_surface,
     &  vi_HPBL_surface,
     &  vi_LAND_surface,
     &  vi_ICEC_surface,
     &  vi_PRATE_surface, 
     &  vi_UFLX_surface,
     &  vi_VFLX_surface,
     &  vi_delz


C  ! for output

      real*4 :: lat(NY), lon(NX)
c     integer :: time_in(NT)
      real*8 :: time_in(NT)
c     real*8 :: timed(NT)
      real, dimension(NX,NY,NT) ::
     &  DLWRF,
     &  ULWRF,
     &  DSWRF,
     &  vbdsf_ave,
     &  vddsf_ave,
     &  nbdsf_ave,
     &  nddsf_ave,
     &  dusfc,
     &  dvsfc,
     &  shtfl_ave,
     &  lhtfl_ave,
     &  totprcp_ave,
     &  u10m,
     &  v10m,
     &  hgt_hyblev1,
     &  psurf,
     &  tmp_hyblev1,
     &  spfh_hyblev1,
     &  ugrd_hyblev1,
     &  vgrd_hyblev1,
     &  q2m,
     &  t2m,
     &  slmsksfc,
     &  pres_hyblev1,
     &  precp,
     &  fprecp, 
     &  icecsfc

      integer :: mcid,outdim(NDIM), outdimx,outdimy,outdimt,
     &  vi_t_dim,
     &  vi_y_dim,
     &  vi_x_dim,
     &  vi_lon,
     &  vi_lat,
     &  vi_time,
     &  vi_DLWRF,
     &  vi_ULWRF,
     &  vi_DSWRF,
     &  vi_vbdsf_ave,
     &  vi_vddsf_ave,
     &  vi_nbdsf_ave,
     &  vi_nddsf_ave,
     &  vi_dusfc,
     &  vi_dvsfc,
     &  vi_shtfl_ave,
     &  vi_lhtfl_ave,
     &  vi_totprcp_ave,
     &  vi_u10m,
     &  vi_v10m,
     &  vi_hgt_hyblev1,
     &  vi_psurf,
     &  vi_tmp_hyblev1,
     &  vi_spfh_hyblev1,
     &  vi_ugrd_hyblev1,
     &  vi_vgrd_hyblev1,
     &  vi_q2m,
     &  vi_t2m,
     &  vi_slmsksfc,
     &  vi_pres_hyblev1,
     &  vi_precp,
     &  vi_fprecp,
     &  vi_icecsfc

C----------------------------------
      nerr = nf_open(FILE_NAME, NF_NOWRITE, ncid)
      if (nerr .ne. nf_noerr) call handle_err(nerr)

C  ! Get the varid of the data variable, based on its name.

      nerr = nf_inq_varid(ncid , "latitude" ,vi_latitude )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "longitude" ,vi_longitude )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "time" ,vi_time_in )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "HGT_1hybridlevel" ,vi_HGT_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TMP_1hybridlevel" ,vi_TMP_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SPFH_1hybridlevel" ,vi_SPFH_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "UGRD_1hybridlevel" ,vi_UGRD_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "VGRD_1hybridlevel" ,vi_VGRD_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "PRES_surface" ,vi_PRES_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "HGT_surface" ,vi_HGT_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TMP_surface" ,vi_TMP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TSOIL_0M0D1mbelowground" ,vi_TSOIL_0M0D1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILW_0M0D1mbelowground" ,vi_SOILW_0M0D1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILL_0M0D1mbelowground" ,vi_SOILL_0M0D1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TSOIL_0D1M0D4mbelowground" ,vi_TSOIL_0D1M0D4mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILW_0D1M0D4mbelowground" ,vi_SOILW_0D1M0D4mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILL_0D1M0D4mbelowground" ,vi_SOILL_0D1M0D4mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TSOIL_0D4M1mbelowground" ,vi_TSOIL_0D4M1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILW_0D4M1mbelowground" ,vi_SOILW_0D4M1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILL_0D4M1mbelowground" ,vi_SOILL_0D4M1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TSOIL_1M2mbelowground" ,vi_TSOIL_1M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILW_1M2mbelowground" ,vi_SOILW_1M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILL_1M2mbelowground" ,vi_SOILL_1M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOILM_0M2mbelowground" ,vi_SOILM_0M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "CNWAT_surface" ,vi_CNWAT_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "WEASD_surface" ,vi_WEASD_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SNOD_surface" ,vi_SNOD_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "PEVPR_surface" ,vi_PEVPR_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "ICETK_surface" ,vi_ICETK_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "ACOND_surface" ,vi_ACOND_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TMP_2maboveground" ,vi_TMP_2maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SPFH_2maboveground" ,vi_SPFH_2maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "UGRD_10maboveground" ,vi_UGRD_10maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "VGRD_10maboveground" ,vi_VGRD_10maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "CPOFP_surface" ,vi_CPOFP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SFCR_surface" ,vi_SFCR_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "FRICV_surface" ,vi_FRICV_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SHTFL_surface" ,vi_SHTFL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "LHTFL_surface" ,vi_LHTFL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SFEXC_surface" ,vi_SFEXC_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "VEG_surface" ,vi_VEG_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "GFLUX_surface" ,vi_GFLUX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "VGTYP_surface" ,vi_VGTYP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SOTYP_surface" ,vi_SOTYP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SLTYP_surface" ,vi_SLTYP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "WILT_surface" ,vi_WILT_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "FLDCP_surface" ,vi_FLDCP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "SUNSD_surface" ,vi_SUNSD_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "PWAT_entireatmosphere_consideredasasinglelayer_" 
     &,vi_PWAT_entireatmosphere_consideredasasinglelayer_ )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "TCDC_convectivecloudlayer" ,vi_TCDC_convectivecloudlayer )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "DSWRF_surface" ,vi_DSWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "DLWRF_surface" ,vi_DLWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "USWRF_surface" ,vi_USWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "ULWRF_surface" ,vi_ULWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "HPBL_surface" ,vi_HPBL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "LAND_surface" ,vi_LAND_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "ICEC_surface" ,vi_ICEC_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "PRATE_surface" ,vi_PRATE_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "UFLX_surface" ,vi_UFLX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "VFLX_surface" ,vi_VFLX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_inq_varid(ncid , "delz" ,vi_delz )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
  

C  ! Read the data.
C! call check( nf90_get_var(ncid, varid, data_in) )


      nerr = nf_get_var_real(ncid , vi_latitude, latitude )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_longitude, longitude )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
c     nerr = nf_get_var_int(ncid , vi_time_in, time_in )
      nerr = nf_get_var_double(ncid , vi_time_in, time_in )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_HGT_1hybridlevel, HGT_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TMP_1hybridlevel, TMP_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SPFH_1hybridlevel, SPFH_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_UGRD_1hybridlevel, UGRD_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_VGRD_1hybridlevel, VGRD_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_PRES_surface, PRES_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_HGT_surface, HGT_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TMP_surface, TMP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TSOIL_0M0D1mbelowground, TSOIL_0M0D1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILW_0M0D1mbelowground, SOILW_0M0D1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILL_0M0D1mbelowground, SOILL_0M0D1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TSOIL_0D1M0D4mbelowground, TSOIL_0D1M0D4mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILW_0D1M0D4mbelowground, SOILW_0D1M0D4mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILL_0D1M0D4mbelowground, SOILL_0D1M0D4mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TSOIL_0D4M1mbelowground, TSOIL_0D4M1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILW_0D4M1mbelowground, SOILW_0D4M1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILL_0D4M1mbelowground, SOILL_0D4M1mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TSOIL_1M2mbelowground, TSOIL_1M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILW_1M2mbelowground, SOILW_1M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILL_1M2mbelowground, SOILL_1M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOILM_0M2mbelowground, SOILM_0M2mbelowground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_CNWAT_surface, CNWAT_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_WEASD_surface, WEASD_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SNOD_surface, SNOD_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_PEVPR_surface, PEVPR_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_ICETK_surface, ICETK_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_ACOND_surface, ACOND_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TMP_2maboveground, TMP_2maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SPFH_2maboveground, SPFH_2maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_UGRD_10maboveground, UGRD_10maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_VGRD_10maboveground, VGRD_10maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_CPOFP_surface, CPOFP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SFCR_surface, SFCR_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_FRICV_surface, FRICV_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SHTFL_surface, SHTFL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_LHTFL_surface, LHTFL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SFEXC_surface, SFEXC_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_VEG_surface, VEG_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_GFLUX_surface, GFLUX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_VGTYP_surface, VGTYP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SOTYP_surface, SOTYP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SLTYP_surface, SLTYP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_WILT_surface, WILT_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_FLDCP_surface, FLDCP_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_SUNSD_surface, SUNSD_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_PWAT_entireatmosphere_consideredasasinglelayer_, 
     &PWAT_entireatmosphere_consideredasasinglelayer_ )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_TCDC_convectivecloudlayer, TCDC_convectivecloudlayer )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_DSWRF_surface, DSWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_DLWRF_surface, DLWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_USWRF_surface, USWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_ULWRF_surface, ULWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_HPBL_surface, HPBL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_LAND_surface, LAND_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_ICEC_surface, ICEC_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_PRATE_surface, PRATE_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_UFLX_surface, UFLX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_VFLX_surface, VFLX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_get_var_real(ncid , vi_delz, delz )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
  


 
C------------------------------ 
      nerr = nf_close(ncid)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
C------------------------------ 

      print *, "TIME= ", time_in
c     print *, "Longitude= ", longitude
      print *,"*** SUCCESS reading example file ", FILE_NAME, "! "

      nerr = nf_create(FOUT_NAME, NF_CLOBBER, mcid)
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      time_out = 1.0*time_in
C: set dim
      nerr = nf_def_dim(mcid, "lon", NX, vi_x_dim)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_dim(mcid, "lat", NY, vi_y_dim)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_dim(mcid, "time",NT,vi_t_dim)
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      outdim(3) = vi_t_dim
      outdim(2) = vi_y_dim
      outdim(1) = vi_x_dim
      outdimt = vi_t_dim
      outdimy = vi_y_dim
      outdimx = vi_x_dim

C: define variables
      nerr = nf_def_var(mcid , "lon" ,NF_REAL, 1, vi_x_dim, vi_lon )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "lat" ,NF_REAL, 1, vi_y_dim, vi_lat )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "time" ,NF_DOUBLE, 1, vi_t_dim, vi_time )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "DLWRF" ,NF_REAL, NDIM, outdim, vi_DLWRF )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "ULWRF" ,NF_REAL, NDIM, outdim, vi_ULWRF )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "DSWRF" ,NF_REAL, NDIM, outdim, vi_DSWRF )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "vbdsf_ave" ,NF_REAL, NDIM, outdim, vi_vbdsf_ave )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "vddsf_ave" ,NF_REAL, NDIM, outdim, vi_vddsf_ave )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "nbdsf_ave" ,NF_REAL, NDIM, outdim, vi_nbdsf_ave )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "nddsf_ave" ,NF_REAL, NDIM, outdim, vi_nddsf_ave )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "dusfc" ,NF_REAL, NDIM, outdim, vi_dusfc )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "dvsfc" ,NF_REAL, NDIM, outdim, vi_dvsfc )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "shtfl_ave" ,NF_REAL, NDIM, outdim, vi_shtfl_ave )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "lhtfl_ave" ,NF_REAL, NDIM, outdim, vi_lhtfl_ave )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "totprcp_ave" ,NF_REAL, NDIM, outdim, vi_totprcp_ave )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "u10m" ,NF_REAL, NDIM, outdim, vi_u10m )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "v10m" ,NF_REAL, NDIM, outdim, vi_v10m )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "hgt_hyblev1" ,NF_REAL, NDIM, outdim, vi_hgt_hyblev1 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "psurf" ,NF_REAL, NDIM, outdim, vi_psurf )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "tmp_hyblev1" ,NF_REAL, NDIM, outdim, vi_tmp_hyblev1 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "spfh_hyblev1" ,NF_REAL, NDIM, outdim, vi_spfh_hyblev1 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "ugrd_hyblev1" ,NF_REAL, NDIM, outdim, vi_ugrd_hyblev1 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "vgrd_hyblev1" ,NF_REAL, NDIM, outdim, vi_vgrd_hyblev1 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "q2m" ,NF_REAL, NDIM, outdim, vi_q2m )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "t2m" ,NF_REAL, NDIM, outdim, vi_t2m )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "slmsksfc" ,NF_REAL, NDIM, outdim, vi_slmsksfc )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "pres_hyblev1" ,NF_REAL, NDIM, outdim, vi_pres_hyblev1 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "precp" ,NF_REAL, NDIM, outdim, vi_precp )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "fprecp" ,NF_REAL, NDIM, outdim, vi_fprecp )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_def_var(mcid , "icecsfc" ,NF_REAL, NDIM, outdim, vi_icecsfc )
      if (nerr .ne. nf_noerr) call handle_err(nerr)



  
C: put attribute
      vi_indx = vi_lon
      STR = "degrees_east"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_lon
      STR = "Longitude"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_lon
      nerr = nf_put_att_text(mcid, vi_indx, "modulo",1," ")
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_lat
      STR = "degrees_north"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_lat
      STR = "Latitude"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_time
!     STR = "seconds since 1970-01-01 00.00.00"
      STR = "seconds since 1970-01-01 00:00:00.0"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_time
      STR = "gregorian"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"calendar",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_time
      STR = "T"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"axis",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_DLWRF
      STR = "W/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_DLWRF
      STR = "surface downward longwave flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_ULWRF 
      STR = "W/m**2" 
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_ULWRF
      STR = "surface upward longwave flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_DSWRF
      STR = "W/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_DSWRF
      STR = "averaged surface downward shortwave flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_vbdsf_ave
      STR = "W/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_vbdsf_ave
      STR = "Visible Beam Downward Solar Flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_vddsf_ave
      STR = "W/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_vddsf_ave
      STR = "Visible Diffuse Downward Solar Flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_nbdsf_ave
      STR = "W/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_nbdsf_ave
      STR = "Near IR Beam Downward Solar Flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_nddsf_ave
      STR = "W/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_nddsf_ave
      STR = "Near IR Diffuse Downward Solar Flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_dusfc
      STR = "N/m**2" 
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_dusfc
      STR = "surface zonal momentum flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_dvsfc
      STR = "N/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_dvsfc
      STR = "surface meridional momentum flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_shtfl_ave
      STR = "w/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_shtfl_ave
      STR = "surface sensible heat flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_lhtfl_ave
      STR = "w/m**2"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_lhtfl_ave
      STR = "surface latent heat flux"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_totprcp_ave
      STR = "kg/m**2/s"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_totprcp_ave
      STR = "surface precipitationrate"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_u10m
      STR = "m/s"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_u10m
      STR = "10 meter u wind"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_v10m
      STR = "m/s"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_v10m
      STR = "10 meter v wind"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_hgt_hyblev1
      STR = "m"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_hgt_hyblev1 
      STR = "layer 1 height"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_psurf
      STR = "Pa"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_psurf
      STR = "surface pressure"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_tmp_hyblev1
      STR = "K"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_tmp_hyblev1 
      STR = "layer 1 temperature"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_spfh_hyblev1
      STR = "kg/kg"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_spfh_hyblev1
      STR = "layer 1 specific humidity"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_ugrd_hyblev1
      STR = "m/s"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_ugrd_hyblev1
      STR = "layer 1 zonal wind"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_vgrd_hyblev1
      STR = "m/s" 
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_vgrd_hyblev1
      STR = "layer 1 meridional wind"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_q2m
      STR = "kg/kg"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_q2m
      STR = "2m specific humidity"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_t2m
      STR = "K"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_t2m 
      STR = "2m temperature"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_slmsksfc
      STR = "numerical"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_slmsksfc
      STR = "sea-land-ice mask (0-sea, 1-land, 2-ice)"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_pres_hyblev1
      STR = "Pa"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_pres_hyblev1
      STR = "layer 1 pressure"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_precp
      STR = "kg/m**2/s"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_precp
      STR = "surface rain precipitation rate"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_fprecp
      STR = "kg/m**2/s"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_fprecp
      STR = "surface snow precipitation rate"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_icecsfc
      STR = "numerical"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"units",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      vi_indx = vi_icecsfc
      STR = "sea-ice coverage/fraction, proportion"
      NSTR = len_trim(STR)
      nerr = nf_put_att_text(mcid, vi_indx,"long_name",NSTR,trim(STR))
      if (nerr .ne. nf_noerr) call handle_err(nerr)

C----------------
      nerr = nf_enddef(mcid)
      if (nerr .ne. nf_noerr) call handle_err(nerr)   
C----------------

 
C: put vaiables

      nerr = nf_put_var_real(mcid , vi_lon, longitude )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_lat, latitude )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_double(mcid , vi_time, time_in )
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      nerr = nf_put_var_real(mcid , vi_DLWRF, DLWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_ULWRF, ULWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_DSWRF, DSWRF_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      VAR2=DSWRF_surface*0.285
      nerr = nf_put_var_real(mcid , vi_vbdsf_ave, VAR2 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      VAR2=DSWRF_surface*0.285
      nerr = nf_put_var_real(mcid , vi_vddsf_ave, VAR2 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      VAR2=DSWRF_surface*0.215
      nerr = nf_put_var_real(mcid , vi_nbdsf_ave, VAR2 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      VAR2=DSWRF_surface*0.215
      nerr = nf_put_var_real(mcid , vi_nddsf_ave, VAR2 )
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      nerr = nf_put_var_real(mcid , vi_dusfc, UFLX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_dvsfc, VFLX_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
C-- bulk method for momentum fluxes
C--------- J/Kg/K
      Rsp = 287.058
C     
C     do k=1,NT
C       do j=1,NY
C         do i=1,NX
C---------  rho_air: Kg/m^3
C           rho_air=PRES_surface(i,j,k)/(Rsp*TMP_surface(i,j,k))
C           UVS=SQRT(UGRD_10maboveground(i,j,k)**2+VGRD_10maboveground(i,j,k)**2)
C           UUS=UGRD_10maboveground(i,j,k)*UVS 
C           VVS=VGRD_10maboveground(i,j,k)*UVS
C           if (UVS > 3.0) then
C             Cdrag=0.61+0.063*UVS
C           else
C             Cdrag=0.61+0.57/UVS
C           endif 
C           dusfc(i,j,k)=rho_air*Cdrag*UUS 
C           dvsfc(i,j,k)=rho_air*Cdrag*VVS 
C         enddo
C       enddo
C     enddo
C     nerr = nf_put_var_real(mcid , vi_dusfc, dusfc )
C     if (nerr .ne. nf_noerr) call handle_err(nerr)
C     nerr = nf_put_var_real(mcid , vi_dvsfc, dvsfc )
C     if (nerr .ne. nf_noerr) call handle_err(nerr)

      nerr = nf_put_var_real(mcid , vi_shtfl_ave, SHTFL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_lhtfl_ave, LHTFL_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
C     nerr = nf_put_var_real(mcid , vi_totprcp_ave,
C    &PWAT_entireatmosphere_consideredasasinglelayer_ )
      nerr = nf_put_var_real(mcid , vi_totprcp_ave, PRATE_surface)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_u10m, UGRD_10maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_v10m, VGRD_10maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_psurf, PRES_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_tmp_hyblev1, TMP_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_spfh_hyblev1, SPFH_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_ugrd_hyblev1, UGRD_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_vgrd_hyblev1, VGRD_1hybridlevel )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_q2m, SPFH_2maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_t2m, TMP_2maboveground )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_slmsksfc, LAND_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
C--------------------------------------------------
      do k=1,NT
        do j=1,NY
          jinv = NY- (j - 1)
          do i=1,NX
            vtmp(i,j,k) = -1.0*delz(i,jinv,k)             
          enddo
        enddo
      enddo
C--------------------------------------------------
      nerr = nf_put_var_real(mcid , vi_hgt_hyblev1, vtmp)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
C--------------------------------------------------
      coef = 1.0
      g0 = 9.80665
C--------- Specific Gas Constant for dry air : J/Kg/K
      Rsp = 287.058
      do k=1,NT
        do j=1,NY
          do i=1,NX
c           TCel = TMP_surface(i,j,k)-273.15
            TCel = TMP_2maboveground(i,j,k)-273.15
            if (TCel >= 0.0) then
              coef = 1.0
            else if (TCel < -15.0) then
              coef = 0.0
            else
              coef = (TCel + 15.0)/15.0
            endif
c           PRAT_tot=PWAT_entireatmosphere_consideredasasinglelayer_(i,j,k)
            PRATE_surf=PRATE_surface(i,j,k)
            precp(i,j,k) = coef * PRATE_surf
            fprecp(i,j,k) = (1.0 - coef) * PRATE_surf
C------------ GFS has correct HGT_1HYBRIDLEVEL
cc           if (abs(HGT_1HYBRIDLEVEL(i,j,k)) < 1.0e10 .and. LAND_surface(i,j,k) == 1) then
c            if (abs(HGT_1HYBRIDLEVEL(i,j,k)) < 1.0e11 .and. LAND_surface(i,j,k) == 1) then
c              Hn = HGT_1HYBRIDLEVEL(i,j,k)
c            else
cc             Hn = 20.0
cc             Hn = HGT_surface(i,j,k) + 1.974e-4 * PRES_surface(i,j,k)
c              Hn = 1.974e-4 * PRES_surface(i,j,k)
c            endif
c            hgt_hyblev1(i,j,k) = Hn
C------------
C           term1 = -1.0*g0*Hn/(Rsp*TMP_1HYBRIDLEVEL(i,j,k))
            term1 = -1.0*g0*vtmp(i,j,k)/(Rsp*TMP_1HYBRIDLEVEL(i,j,k))
            term2 = exp(term1)
            pres_hyblev1(i,j,k)=PRES_SURFACE(i,j,k)*term2  
          enddo
        enddo
      enddo
C--------------------------------------------------
C     nerr = nf_put_var_real(mcid , vi_hgt_hyblev1, hgt_hyblev1 )
c     nerr = nf_put_var_real(mcid , vi_hgt_hyblev1, HGT_1HYBRIDLEVEL )
c     if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_pres_hyblev1, pres_hyblev1)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_precp, precp)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_fprecp, fprecp)
      if (nerr .ne. nf_noerr) call handle_err(nerr)
      nerr = nf_put_var_real(mcid , vi_icecsfc, ICEC_surface )
      if (nerr .ne. nf_noerr) call handle_err(nerr)
C: close 
      nerr = nf_close(mcid)
      if (nerr .ne. nf_noerr) call handle_err(nerr)

      end

C=======================================================
      subroutine handle_err(errcode)
      implicit none
      include 'netcdf.inc'
      integer errcode

      print *, 'Error: ', nf_strerror(errcode)
      stop 2
      end


C=======================================================
      subroutine convolt(NX,NY,NT,invar,outvar,conmuti)
      implicit none
      integer I,J,K,NX,NY,NT
      real conmuti
      real  :: invar(NT,NY,NX), outvar(NX,NY,NT)
C: set output variables

      do k=1,NT
        do j=1,NY
          do i=1,NX
            outvar(i,j,k) = invar(k,j,i) * conmuti
          enddo
        enddo
      enddo
      return
      end


