#!/usr/bin/env bash
# Downloads 6 weeks (2024-06-01 .. 2024-07-12) of 20 Hz bottom-pressure raw files
# for two co-located OOI Axial Seamount BOTPT instruments.
set -u
B=https://rawdata.oceanobservatories.org/files
OUT=/home/drew/DANCR/testdata/raw
declare -A INSTR=( [MJ03F]="RS03CCAL/MJ03F/BOTPTA301|BOTPTA301_10.31.6.5_9338" [MJ03E]="RS03ECAL/MJ03E/BOTPTA302|BOTPTA302_10.31.10.6_9338" )
for site in MJ03F MJ03E; do
  IFS='|' read -r path prefix <<< "${INSTR[$site]}"
  mkdir -p "$OUT/$site"
  for i in $(seq 0 41); do
    d=$(date -u -d "2024-06-01 +$i days" +%Y%m%d)
    ym=$(date -u -d "2024-06-01 +$i days" +%Y/%m)
    f="${prefix}_${d}T0000_UTC.dat"
    dest="$OUT/$site/$f"
    if [ -s "$dest" ] && [ ! -f "$dest.part" ]; then continue; fi
    echo "$(date +%T) GET $site $d"
    touch "$dest.part"
    curl -s -f --retry 3 -o "$dest" "$B/$path/$ym/$f" && rm -f "$dest.part" || echo "FAILED $site $d"
  done
done
echo "DONE $(date +%T)"
