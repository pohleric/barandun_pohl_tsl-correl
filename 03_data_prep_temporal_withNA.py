''''
This file produces the Input files for R

Use combined temporal basis PAR file from 000D
'''
import pandas as pd
import numpy as np

long_period_end = [2018]
short_period_end = [2014]
period_start = [2000]

# ERA5 data includes the 1979 to present data
years_exclude = [1979,
                 1980, 1981, 1982,1983,1984,1985,1986,1987,1988,1989,
                 1990, 1991, 1992,1993,1994,1995,1996,1997,1998,1999,
                 2019, 2020]

dataset_names = ['ECMWF','HAR','CHELSA', 'MOD10CM']
thresh_na_value = 1e+10  # positive and negative fill values of ~ xE+38 // change 2022-04-08

mnt = r'Z:/'

# PAR base file from the map_all_summary.csv files
par_infile = mnt+'data/2d_PTS_pars_temporal_filtered.csv'

# output PAR files for temporal correlation in R - Estimates MARTINA
par_input_outfile_short = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_short.csv'
par_input_outfile_long = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_long.csv'

# output PAR files for temporal correlation in R - Estimates HUGONNET
par_input_outfile_short_h = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_short_hugonnet.csv'
par_input_outfile_long_h = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_long_hugonnet.csv'

aggregation_levels = ['hydro','winter','spring','summer','autumn']
# =================================================================================================== #
df_all = pd.read_csv(par_infile, index_col=0)

# =================================================================================================== #
# subset into X and Y
# THe potential regressor names:
par_clim = [
    "t2",
    "t2m",
    "tas",
    "tp",
    "pr",
    "prcp",
    "scf",
    "scd"
]
# Identify correct columns
# 1) Dataset name
cols_datasets_ind = [i for i,cni in enumerate(df_all.columns) if str(cni).split('_')[0] in dataset_names]
cols_datasets = [df_all.columns[i] for i in cols_datasets_ind]
# 2) Variable names
# cols_datasets = [i for i,cni in enumerate(df_all.columns) if str(cni).split('_')[0] in dataset_names]
cols_ss_datasets_ind = [i for i, cni in enumerate(cols_datasets) if str(cni).split('_')[1] in par_clim]
cols_ss_datasets = [cols_datasets[i] for i in cols_ss_datasets_ind]

# 3) years to exclude (half-periods) ---- NOT FOR TEMPORAL ANALYSIS
# mean value pars are 5 long
# trend value pars are 6 long
cols_ss_keep = []
for cn in cols_ss_datasets:
    cn_split = cn.split('_')[-1]
    # print(cn_split)
    keep = True
    for cnsi in cn_split.split('-'):
        if int(cnsi) in years_exclude:
            keep = False
    if keep:
        cols_ss_keep.append(cn)

cols_ss_keep_par = cols_ss_keep

# 5) Periods
# Subset into:
cols_period1_start = []
cols_period1_end = []
cols_period2_start = []
cols_period2_end = []
# - 2000/2001 to 2014
# - 2000/2001 to 2018
for i,cni in enumerate(cols_ss_keep_par):
    cn_split = cni.split('_')[-1]
    long_p_diff = (long_period_end[0] - period_start[0])
    short_p_diff = (short_period_end[0] - period_start[0])
    if int(cn_split) in period_start:
        cols_period1_start.append(cni)
        cols_period2_start.append(cni)
        try:
            cn_split_end_long = cols_ss_keep_par[i + long_p_diff].split('_')[-1]
            cn_split_end_short = cols_ss_keep_par[i + short_p_diff].split('_')[-1]
            if int(cn_split_end_short) in short_period_end:
                cols_period1_end.append(cols_ss_keep_par[i + short_p_diff])
            if int(cn_split_end_long) in long_period_end:
                cols_period2_end.append(cols_ss_keep_par[i + long_p_diff])
            else:
                cols_period2_end.append(np.nan)
        except:
            cols_period2_end.append(np.nan)

p1_list = [[a, b] for a,b in zip(cols_period1_start,cols_period1_end) if b is not np.nan]
p2_list = [[a, b] for a,b in zip(cols_period2_start,cols_period2_end) if b is not np.nan]

p1_df = pd.concat([df_all.loc[:,str(p1_list_i[0]):str(p1_list_i[1])] for p1_list_i in p1_list],axis=1)
p2_df = pd.concat([df_all.loc[:,str(p2_list_i[0]):str(p2_list_i[1])] for p2_list_i in p2_list],axis=1)

# 2022-04-08 - remove entries with incomplete data (threshold ~+- E+38)
p1_df[p1_df > thresh_na_value] = pd.NA
p1_df[p1_df < (-1 * thresh_na_value)] = pd.NA
p2_df[p2_df > thresh_na_value] = pd.NA
p2_df[p2_df < (-1 * thresh_na_value)] = pd.NA

# =================================================================================================== #

X_pshort = p1_df
X_plong = p2_df

# ESTIMATES MARTINA
Y_long = df_all.loc[:,'2000':'2018']  # drop also the same na rows as for X_xlon
Y_short = df_all.loc[:,'2000':'2014']

# ESTIMATES HUGONNET
Y_long_h = df_all.loc[:,'mb_hug_2000':'mb_hug_2018']
Y_short_h = df_all.loc[:,'mb_hug_2000':'mb_hug_2014']

# =================================================================================================== #
# write output
# Combined Parameter files for processing in R/python
# MARTINA
df_out_long = pd.DataFrame(X_plong)
df_out_long = df_out_long.join(Y_long)

df_out_short = pd.DataFrame(X_pshort)
df_out_short = df_out_short.join(Y_short)

# write:
df_out_long.to_csv(par_input_outfile_long)
df_out_short.to_csv(par_input_outfile_short)

# HUGONNET
df_out_long_h = pd.DataFrame(X_plong)
df_out_long_h = df_out_long_h.join(Y_long_h)

df_out_short_h = pd.DataFrame(X_pshort)
df_out_short_h = df_out_short_h.join(Y_short_h)

# write:
df_out_long_h.to_csv(par_input_outfile_long_h)
df_out_short_h.to_csv(par_input_outfile_short_h)

# =================================================================================================== #
