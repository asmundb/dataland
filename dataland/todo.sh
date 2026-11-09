#!/bin/bash

cd ~/Projects/dataland
. .venv/bin/activate


# output_domains
#metcoop_domain=/lustre/storeB/project/nwp/bris/datasets/aifs-meps-2.5km-2020-2025-6h-v7.zarr
metcoop_domain="/home/asmundb/Projects/dataland/data/output_data/old_metcoop/slope_oro_metcoop_100.nc"
O96_domain=/lustre/storeB/project/nwp/bris/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v2-s2s-predictors.zarr
N320_domain=/lustre/storeB/project/nwp/bris/datasets/aifs-ea-an-oper-0001-mars-n320-1979-202605-6h-v1-seaice.zarr
carra2_domain=/lustre/storeB/users/ivaras/carra2/data/zarr/carra2-2.5km-20230101-20241231-6h.zarr
# fractions

domain_name=metcoop
domain=$metcoop_domain

source_eco=data/input_data/ecosg_final_map.nc
source_ht=data/input_data/new_ht_c_m.nc
source_topo=data/input_data/GMTED2010_30.EHdr.nc
#echo $carra2_domain

#python -m dataland.aggregate_to_grid \
#    --source-file data/input_data/ecosg_final_map.nc\
#    --domain $domain_name \
#    --output data/output_data/fractions_$domain_name.nc \
#    --memory-mb 128 \
#    --point-file $domain \
#    --point-latitude latitude \
#    --point-longitude longitude \
#    --source-var COVER \
#    --mode fractions \
#    --fill-uncovered 0 || exit 1
#
#python -m dataland.aggregate_to_grid \
#    --source-file data/input_data/new_ht_c_m.nc\
#    --domain $domain_name \
#    --output data/output_data/tree_height_$domain_name.nc \
#    --memory-mb 128 \
#    --point-file $domain \
#    --point-latitude latitude \
#    --point-longitude longitude \
#    --source-var H_TREE \
#    --mode mean \
#    --fill-missing-mean 0 \
#    --fill-uncovered 0 || exit 1
#
#
#for radius in 2500 15000 50000; do
#    python -m dataland.aggregate_to_grid \
#        --source-file $source_topo\
#        --domain $domain_name \
#        --output data/output_data/std_oro_${domain_name}_${radius}.nc \
#        --memory-mb 128 \
#        --point-file $domain \
#        --point-latitude latitude \
#        --point-longitude longitude \
#        --source-var orography \
#        --mode std \
#        --radius-m $radius \
#        --fill-uncovered 0 || exit 1
#
#    python -m dataland.aggregate_to_grid \
#        --source-file $source_topo\
#        --domain $domain_name \
#        --output data/output_data/slope_oro_${domain_name}_${radius}.nc \
#        --memory-mb 128 \
#        --point-file $domain \
#        --point-latitude latitude \
#        --point-longitude longitude \
#        --source-var orography \
#        --mode slope \
#        --radius-m $radius \
#        --fill-uncovered 0 || exit 1    
#done
#

anemoi-datasets create dataland/config/anemoi_$domain_name.yaml --overwrite data/zarr_datasets_out/static_forcings_$domain_name.zarr

python -m dataland.plotting data/zarr_datasets_out/static_forcings_$domain_name.zarr  \
    --output-dir figures/zoom \
    --zoom 8 58 15 61 \
    --background-color grey

python -m dataland.plotting data/zarr_datasets_out/static_forcings_$domain_name.zarr  \
    --output-dir figures \
    --background-color grey
