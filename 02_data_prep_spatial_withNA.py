''''
This file produces the Input files for R

Use combined static/spatil basis PAR file from 000C
'''

import pandas as pd

periods = [[2000,2014], [2000,2009],[2006,2014], [2000,2018]]
long_period = [2018,2019]
short_period = [2013,2014]

dataset_names = ['ERA5','HAR30','CHELSA', 'MOD10']

mnt = r'Z:/'
# PAR base file from the map_all_summary.csv files
par_infile = mnt+'data/2d_PTS_pars_static_filtered_k-means_clusters.csv'

# output spatial correlation (all variables for multiple linear regression)
par_outfile_short = mnt+'data/correlation_spatial/2d_PTS_filtered_correlation_spatial_short.csv'
par_outfile_short_morph = mnt+'data/correlation_spatial/2d_PTS_filtered_correlation_spatial_short_morph.csv'
par_outfile_long = mnt+'data/correlation_spatial/2d_PTS_filtered_correlation_spatial_long.csv'
par_outfile_long_morph = mnt+'data/correlation_spatial/2d_PTS_filtered_correlation_spatial_long_morph.csv'

# output PAR files for spatial correlation in R - Estimates MARTINA
par_input_outfile_short = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short.csv'
par_input_outfile_short_morph = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short_morph.csv'
par_input_outfile_long = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long.csv'
par_input_outfile_long_morph = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long_morph.csv'

# output PAR files for spatial correlation in R - Estimates HUGONNET
par_input_outfile_short_h = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short_hugonnet.csv'
par_input_outfile_short_morph_h = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short_morph_hugonnet.csv'
par_input_outfile_long_h = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long_hugonnet.csv'
par_input_outfile_long_morph_h = mnt+'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long_morph_hugonnet.csv'
pars = ['out_mean',
        'out_std',
        'out_slope']

aggregation_levels = ['hydro','winter','spring','summer','autumn']
# remove half time series estimates
exclude_years = [2006,2007,2008, 2009, 2019, 2020]
# =================================================================================================== #
df_all = pd.read_csv(par_infile, index_col=0)

# =================================================================================================== #
# subset into X and Y
# for X use all geomorphic parameters
par_morph = [
    'med. elev',
    'aspect',
    'slope over tongue',
    'area ',  # with space in name
    'asp_cos',
    'asp_sin',
    'lat ',   # with space in name
    'long',
    'carea'
]
par_clim = [
    "t2",
    "t2m",
    "tas",
    "tp",
    "pr",
    "prcp",
    "scf",
    "scd",
    "msnswrf"
    ]
# Identify correct columns
# 1) Dataset name
cols_datasets_ind = [i for i,cni in enumerate(df_all.columns) if str(cni).split('_')[0] in dataset_names]
cols_datasets = [df_all.columns[i] for i in cols_datasets_ind]
# 2) Variable names
cols_ss_datasets_ind = [i for i, cni in enumerate(cols_datasets) if str(cni).split('_')[1] in par_clim]
cols_ss_datasets = [cols_datasets[i] for i in cols_ss_datasets_ind]
# 3) years to exclude (half-periods)
# mean value pars are 5 long
# trend value pars are 6 long
cols_ss_keep = []
for cn in cols_ss_datasets:
    cn_split = cn.split('_')[-1]
    # print(cn_split)
    keep = True
    for cnsi in cn_split.split('-'):
        if int(cnsi) in exclude_years:
            keep = False
    if keep:
        cols_ss_keep.append(cn)

# 4) Correct statistics
cols_ss_keep_par = []
for cn in cols_ss_keep:
    cntmp = [cn for parsi in pars if parsi in cn]
    if len(cntmp) > 0:
        cols_ss_keep_par.append(cn)

# 5) Periods
# Subset into:
cols_period1 = []
cols_period2 = []
# - 2000/2001 to 2014
# - 2000/2001 to 2018
for cni in cols_ss_keep_par:
    cn_split = cni.split('_')[-1]
    cn_split_end = cn_split.split('-')[-1]
    if int(cn_split_end) in long_period:
        cols_period1.append(cni)
    elif int(cn_split_end) in short_period:
        cols_period2.append(cni)

# =================================================================================================== #
X_plong = df_all[cols_period1]
X_pshort = df_all[cols_period2]
# add the morphological parameters
X_morph = df_all[par_morph]

X_plong_morph = X_plong.join(X_morph)
X_pshort_morph = X_pshort.join(X_morph)

# ESTIMATES MARTINA
Y_long = df_all['mb']
Y_short_df = df_all.loc[:,'2000':'2014']
Y_short = Y_short_df.mean(axis=1)

# ESTIMATES HUGONNET
Y_long_h = df_all['mb_hug_2000-2018']
Y_short_h = df_all['mb_hug_2000-2014']


# =================================================================================================== #
# write output
# Combined Parameter files for processing in R/python
# MARTINA
df_out_long = pd.DataFrame(X_plong)
df_out_long['mb'] = Y_long

df_out_long_morph = pd.DataFrame(X_plong_morph)
df_out_long_morph['mb'] = Y_long

df_out_short = pd.DataFrame(X_pshort)
df_out_short['mb'] = Y_short

df_out_short_morph = pd.DataFrame(X_pshort_morph)
df_out_short_morph['mb'] = Y_short

# write:
df_out_long.to_csv(par_input_outfile_long)
df_out_long_morph.to_csv(par_input_outfile_long_morph)
df_out_short.to_csv(par_input_outfile_short)
df_out_short_morph.to_csv(par_input_outfile_short_morph)

# HUGONNET
df_out_long_h = pd.DataFrame(X_plong)
df_out_long_h['mb'] = Y_long_h

df_out_long_morph_h = pd.DataFrame(X_plong_morph)
df_out_long_morph_h['mb'] = Y_long_h

df_out_short_h = pd.DataFrame(X_pshort)
df_out_short_h['mb'] = Y_short_h

df_out_short_morph_h = pd.DataFrame(X_pshort_morph)
df_out_short_morph_h['mb'] = Y_short_h
# write:
df_out_long_h.to_csv(par_input_outfile_long_h)
df_out_long_morph_h.to_csv(par_input_outfile_long_morph_h)
df_out_short_h.to_csv(par_input_outfile_short_h)
df_out_short_morph_h.to_csv(par_input_outfile_short_morph_h)
