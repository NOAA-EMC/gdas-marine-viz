help([[
Load environment for running EVA.
]])

local pkgName    = myModuleName()
local pkgVersion = myModuleVersion()
local pkgNameVer = myModuleFullName()

conflict(pkgName)

-- prepend_path("MODULEPATH", "/contrib/spack-stack//spack-stack-1.6.0/envs/unified-env-rocky8/install/modulefiles/Core")
prepend_path("MODULEPATH", '/contrib/spack-stack/spack-stack-1.9.2/envs/ue-oneapi-2024.2.1/install/modulefiles/Core')
-- load("stack-intel/2021.5.0")
-- load("python/3.10.13")
-- load("python/3.11")
-- load("proj/9.2.1")

load("stack-oneapi/2024.2.1")
load("stack-intel-oneapi-mpi/2021.13")
load("intel-oneapi-mkl/2024.2.1")
load("stack-python/3.11.7")


load("py-jinja2/3.1.4")
load("py-pyyaml/6.0.2")
load("py-numpy/1.26.4")
load("py-netcdf4/1.7.1.post2")
load("py-matplotlib/3.7.4")
load("py-xarray/2024.7.0")
load("py-cartopy/0.24.1") -- needs stack-oneapi/2024.2.1 
-- load("eva/1.0.0")


-- local pyenvpath = "/scratch1/NCEPDEV/da/python/envs/"
-- local pyenvname = "eva"

-- local pyenvactivate = pathJoin(pyenvpath, pyenvname, "bin/activate")
-- if (mode() == "load") then
--   local activate_cmd = "source "..pyenvactivate
--   execute{cmd=activate_cmd, modeA={"load"}}
-- else
--   if (mode() == "unload") then
--     local deactivate_cmd = "deactivate"
--     execute{cmd=deactivate_cmd, modeA={"unload"}}
--   end
-- end

whatis("Name: ".. pkgName)
whatis("Version: ".. pkgVersion)
whatis("Category: EVA")
whatis("Description: Load all libraries needed for EVA")
