curl http://data.yotahub.com/2024-8/datahub-2024-08-vibration_unbalance.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns vibe_unbal
