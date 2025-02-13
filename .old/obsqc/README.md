Use the following api to check missing files or plotting number of obs and min/max value:
python godas_obssanitycheck.py -t adt -p c2 j2 j3 sa 3a 3b -s 20190201 -e 20190401 --step P1D
It requires obs_type, like adt and platforms, such as j2, j3 etc., also start and end date, default step is P1D

To create a report on  inventory, use:
    a. for P1D database
        python godas_inventory_barplot.py -s 20150101 -e 20220401 -y obs_inventory_p1d.yaml
    b. for PT10M database
        python godas_inventory_barplot.py -s 201901010000 -e 201912310000 -y obs_inventory_pt10m.yaml
Put the obs_type and obs dir in the yaml file


