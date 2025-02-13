#!/bin/csh

#-- PRATE,UFLX,VFLX are from the f006 fcst of the run before 6 hour.
#set hprate = f006

# Set dates and doings
#=====================
set gsdate = 20220315
set gedate = 20220315

set csdate = 20220315
set cedate = 20220315 

set do_get = 1
set do_con = 1
#=====================

if ($do_get == 1) then
 #csh ./get_gfs_beta4.csh $gsdate $gedate
  csh ./get_gfs_beta4_gdas.csh $gsdate $gedate
endif

if ($do_con == 1) then
  csh ./conv_gfs2datm.fort_beta4.csh $csdate $cedate
endif

