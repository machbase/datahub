curl http://data.yotahub.com/2024-9/datahub-2024-09-Appliances-Energy.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns appliance_energy
