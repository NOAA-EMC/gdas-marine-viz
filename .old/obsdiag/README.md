# `obsdiag/`
Scripts used to diagnose in obs space.

- **godas_obssanitycheck.py**

Basic checks of observation data files (obs count, min/max, ...)

Usage example:
```console
python godas_obssanitycheck.py -f adt_c2 -s 20190101 -e 20190321
```

- **godas_oceanview.py**

Back by popular demand. Does require a few packages to be installed on orion, coding is ugly and
probably needs some serious refactoring/cleaning. Provided as is for now

TODO (Guillaume): Refactor and cleanup

Usage:
```console
godas_oceanview.py -i *temp_profile*.nc
```
where `*temp_profile*.nc` is a list of output ioda files from the soca variational application.
