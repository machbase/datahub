curl http://data.yotahub.com/2024-11/datahub-2024-11-Pump-Sensor.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns pump
