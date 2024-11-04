curl http://data.yotahub.com/2024-14/datahub-2024-14-San-Diego-Daily-Weather.csv.gz | machbase-neo shell import --input - --compress gzip --header --method append --timeformat ns san_diego_weather
