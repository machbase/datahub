curl http://data.yotahub.com/2024-6/datahub-2024-06-elec-transformer.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns elec_trans
