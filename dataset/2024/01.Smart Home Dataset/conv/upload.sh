curl http://data.yotahub.com/2024-1/datahub-2024-1-home.csv.gz | machbase-neo shell import --input -  --compress gzip --header --method append --timeformat ns home 