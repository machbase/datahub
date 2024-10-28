curl http://data.yotahub.com/2024-12/datahub-2024-12-Gold-Price.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns gold
