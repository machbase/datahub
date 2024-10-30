curl http://data.yotahub.com/2024-13/datahub-2024-13-India-Air-Quality.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns india_air_quality
