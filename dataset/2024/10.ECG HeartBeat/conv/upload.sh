curl http://data.yotahub.com/2024-10/datahub-2024-10-ECG-HeartBeat.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns ecg
