"""
TEMPORAL
Correlation analysis - using always a combination of two regressors.
All glaciers
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy
import statsmodels.formula.api as smf
from itertools import combinations


def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


mnt = r'Z:/'

par_in_short = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_short.csv'
par_in_long = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_long.csv'

# output PAR files for temporal correlation in R - Estimates HUGONNET
par_in_short_h = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_short_hugonnet.csv'
par_in_long_h = mnt+'data/correlation_temporal/2d_PTS_filtered_corInput_temporal_long_hugonnet.csv'

DIR_OUT = mnt + 'data/correlation_temporal_results/'

thresh_na_value = 1e+10  # added the removal in the pre file (001B ... correlation temporal prepForR)

# ----  USE ALL VARIABLES ---- #
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

aggregation_levels = ['hydro', 'winter', 'spring', 'summer', 'autumn']
dataset_names = ['ECMWF', 'HAR', 'CHELSA', 'MOD10CM']

period_short = [2000,2014]
period_long = [2000,2018]

# --------- Spatial correlation --------- #
# output arrays n*m ,n=glaciers, m=variables
# - p-value
# - R2

# ============================== VARIABLES ============================== #


def compute_and_write(df_name, df_out_name_base, clim_or_morph, var_to_use, temporal=False, mb_ref="Martina", period=period_short):
    df = pd.read_csv(df_name, index_col=0)
    df.index.name = 'RGI code'  # make nice and neat
    df_ind = copy(df)

    # filter out variables
    if clim_or_morph == 'climateOnly':
        ind_ss = [cni for cni in df_ind.columns if (cni.split('_')[0] in dataset_names)]
        ind_ss = [cni for cni in ind_ss if (cni.split('_')[1] in var_to_use)]
        ind_ss = [cni for cni in ind_ss if (cni.split('_')[2] in aggregation_levels)]
        df_ind = df_ind[ind_ss]
    elif clim_or_morph == 'morphOnly':
        ind_ss = [cni for cni in df_ind.columns if cni in var_to_use]
        df_ind = df_ind[ind_ss]
    else:
        raise ValueError('Specify "var_to_use" to be either of the two:\n "climateOnly" \n "morphOnly"')

    # input MB
    if temporal:
        if mb_ref == "Martina":
            cns = df.columns.str.split('_')
            single_colnames = np.array([int(cni[0]) for cni in cns if len(cni) == 1])
            start_ = single_colnames[np.where(single_colnames == period[0])[0]][0]
            end_ = single_colnames[np.where(single_colnames == period[1])[0]][0]
            df_dep = df.loc[:, str(start_):str(end_)]
        if mb_ref == "Hugonnet":
            df_dep = df.iloc[:,df.columns.str.match('mb_hug')]
            # check if correct time
            cns = df_dep.columns.str.split('_')
            single_colnames = np.array([int(cni[2]) for cni in cns if len(cni) == 3])
            start_ = single_colnames[np.where(single_colnames == period[0])[0]][0]
            end_ = single_colnames[np.where(single_colnames == period[1])[0]][0]
            df_dep = df.loc[:, 'mb_hug_'+str(start_):'mb_hug_'+str(end_)]
    else:
        raise ValueError('Cannot find the MB values in the data.frame!')

    vars_ind = df_ind.columns.values

    if temporal:
        # Split column names to get the DATASET, VARIABLE, and AGGREGATION
        vars_ind_tmp = []
        for vars_indi in vars_ind:
            struq = vars_indi.split('_')[0] +'_'+ vars_indi.split('_')[1] +'_'+ vars_indi.split('_')[2]
            vars_ind_tmp.append(struq)
        vars_ind_uq = np.unique(vars_ind_tmp)

    # Corr mat each glacier - all variable combinations
    comb0 = []
    if not temporal:
        a = combinations(vars_ind, 2)
    elif temporal:
        a = combinations(vars_ind_uq, 2)
    else:
        raise ValueError('Something wrong with the temporal argument!')

    for t in a:
        comb0.append(t)
    combs = [ci[0] + ';' + ci[1] for ci in comb0 if ci[0] != ci[1] and (ci[0].split('_')[0] == ci[1].split('_')[0] or ci[0].split('_')[0] =='MOD10' or  ci[1].split('_')[0] =='MOD10')]

    out_cor_glac = pd.DataFrame(columns=combs, index=df.index)
    out_pval_glac = copy(out_cor_glac)
    out_slope_glac = copy(out_cor_glac)

    vr_ix1 = {}
    vr_ix2 = {}
    for c0, c1 in tqdm(comb0):
        ind1 = [c0 in vars_ind_i for vars_ind_i in vars_ind]
        ind2 = [c1 in vars_ind_i for vars_ind_i in vars_ind]
        ind_comb1 = ind1
        ind_comb2 = ind2
        tmp_str1 = c0
        tmp_str2 = c1
        vr_ix1[tmp_str1] = ind_comb1
        vr_ix2[tmp_str2] = ind_comb2

    vars_ind_unq_2var = combs
    # if not temporal:
    #     y = df_dep.to_numpy().reshape((-1, 1))
    # elif temporal:
    #     y = df_dep.T.to_numpy()

    for gl in tqdm(out_cor_glac.index):
        isub = df_ind.loc[gl]
        dsub = df_dep.loc[gl]
        # Use one variable at a time
        for vars_i in vars_ind_unq_2var:
            vr1, vr2 = vars_i.split(';')
            ind1 = vr_ix1[vr1]
            ind2 = vr_ix2[vr2]
            x1 = isub[ind1]
            x2 = isub[ind2]
            # clean data - necessary due to missing values of HAR
            # threshold of "5" as too many NAs
            if np.isnan(x2.values).sum() > 5:
                continue
            if np.isnan(x1.values).sum() > 5:
                continue
            # clean NA threshold data
            if any(x1 > thresh_na_value):
                ind_na = np.where(x1 > thresh_na_value)[0]
                x1.iloc[ind_na] = np.nan
            if any(x1 < (-1 * thresh_na_value)):
                ind_na = np.where(x1 < (-1 * thresh_na_value))[0]
                x1.iloc[ind_na] = np.nan
            if any(x2 > thresh_na_value):
                ind_na = np.where(x2 > thresh_na_value)[0]
                x2.iloc[ind_na] = np.nan
            if any(x2 < (-1 * thresh_na_value)):
                ind_na = np.where(x2 < (-1 * thresh_na_value))[0]
                x2.iloc[ind_na] = np.nan

            # which years are available
            years_ind_ix1 = [int(x_ind_i[-1]) for x_ind_i in x1.index.str.split('_')]
            years_ind_ix2 = [int(x_ind_i[-1]) for x_ind_i in x2.index.str.split('_')]
            years_dep_ix = [int(dsub_i.split('_')[-1]) for dsub_i in dsub.index]

            # make sure both independent variables have the same years
            years_ind_ix = [years_ind_ix_i for years_ind_ix_i in years_ind_ix1 if years_ind_ix_i in years_ind_ix2]
            tmp_ind1 = [x1i for x1i in x1.index if int(x1i.split('_')[-1]) in years_ind_ix]
            x1s = x1[tmp_ind1]
            tmp_ind2 = [x2i for x2i in x2.index if int(x2i.split('_')[-1]) in years_ind_ix]
            x2s = x2[tmp_ind2]

            # choose MB values according to years available
            tmp_match_dep_ix = [years_dep_ix.index(x) for x in years_ind_ix]
            y = dsub.iloc[tmp_match_dep_ix]

            # remove Nan
            nan_ix = np.isfinite([x1s, x2s, y]).all(axis=0)
            # X = np.vstack([x1s, x2s])[:, nan_ix]  # take x1,x2 individually instead
            Y = y[nan_ix]

            dat = pd.DataFrame(np.vstack((Y, x1s[nan_ix], x2s[nan_ix])).T, columns=['mb', vr1, vr2])
            # corr
            fit = smf.ols('mb ~ %s + %s' % (vr1, vr2), data=dat).fit()
            slopes = fit.params[1:3].values
            slopes_str = ";".join([str(slope_i) for slope_i in slopes])
            pvalues = fit.pvalues[1:3].values
            pvalues_str = ";".join([str(pval_i) for pval_i in pvalues])
            rsquared = fit.rsquared
            if rsquared < 0:
                print('R2 < 0 encountered between: \n - %s \n - %s '%(vr1, vr2))

            # write to matrix
            out_slope_glac.loc[gl, vars_i] = slopes_str  # [sli for sli in slopes]
            out_cor_glac.loc[gl, vars_i] = rsquared
            out_pval_glac.loc[gl, vars_i] = pvalues_str

    out_cor_glac.to_csv(DIR_OUT + df_out_name_base + 'cor_2var_temporal_' + clim_or_morph + '.csv')
    out_slope_glac.to_csv(DIR_OUT + df_out_name_base + 'slope_2var_temporal_' + clim_or_morph + '.csv')
    out_pval_glac.to_csv(DIR_OUT + df_out_name_base + 'pval_2var_temporal_' + clim_or_morph + '.csv')

    out_cor_glac.to_pickle(DIR_OUT + df_out_name_base + 'cor_2var_temporal_' + clim_or_morph + '.pickle')
    out_slope_glac.to_pickle(DIR_OUT + df_out_name_base + 'slope_2var_temporal_' + clim_or_morph + '.pickle')
    out_pval_glac.to_pickle(DIR_OUT + df_out_name_base + 'pval_2var_temporal_' + clim_or_morph + '.pickle')

    # ------- split matrices into two --------- #
    # P-VAL #
    out_pval_glac1 = copy(out_pval_glac)
    out_pval_glac2 = copy(out_pval_glac)
    for j in tqdm(range(out_pval_glac.shape[1])):
        tmp_vars = out_pval_glac.iloc[:, j].str.split(';', 1).tolist()
        tmp_vars = [[np.nan, np.nan] if isinstance(tmp_vars_i, float) else tmp_vars_i for tmp_vars_i in tmp_vars]
        ab = pd.DataFrame(tmp_vars, columns=['first', 'Last']).values
        out_pval_glac1.iloc[:, j] = ab[:, 0]
        out_pval_glac2.iloc[:, j] = ab[:, 1]

    out_pval_glac1 = out_pval_glac1.astype(float)
    out_pval_glac2 = out_pval_glac2.astype(float)
    out_pval_glac1.columns = [sti.split(';')[0] for sti in out_pval_glac1.columns]
    out_pval_glac2.columns = [sti.split(';')[1] for sti in out_pval_glac2.columns]
    out_pval_glac1.to_csv(DIR_OUT + df_out_name_base + 'pval_2var(var1)_temporal_' + clim_or_morph + '.csv')
    out_pval_glac1.to_pickle(DIR_OUT + df_out_name_base + 'pval_2var(var1)_temporal_' + clim_or_morph + '.pickle')
    out_pval_glac2.to_csv(DIR_OUT + df_out_name_base + 'pval_2var(var2)_temporal_' + clim_or_morph + '.csv')
    out_pval_glac2.to_pickle(DIR_OUT + df_out_name_base + 'pval_2var(var2)_temporal_' + clim_or_morph + '.pickle')

    # SLOPE #
    out_slope_glac1 = copy(out_slope_glac)
    out_slope_glac2 = copy(out_slope_glac)
    for j in tqdm(range(out_slope_glac.shape[1])):
        tmp_vars = out_slope_glac.iloc[:, j].str.split(';', 1).tolist()
        tmp_vars = [[np.nan, np.nan] if isinstance(tmp_vars_i, float) else tmp_vars_i for tmp_vars_i in tmp_vars]
        ab = pd.DataFrame(tmp_vars, columns=['first', 'Last']).values
        out_slope_glac1.iloc[:, j] = ab[:, 0]
        out_slope_glac2.iloc[:, j] = ab[:, 1]

    out_slope_glac1 = out_slope_glac1.astype(float)
    out_slope_glac2 = out_slope_glac2.astype(float)
    out_slope_glac1.columns = [sti.split(';')[0] for sti in out_slope_glac1.columns]
    out_slope_glac2.columns = [sti.split(';')[1] for sti in out_slope_glac2.columns]
    out_slope_glac1.to_csv(DIR_OUT + df_out_name_base + 'slope_2var(var1)_temporal_' + clim_or_morph + '.csv')
    out_slope_glac1.to_pickle(DIR_OUT + df_out_name_base + 'slope_2var(var1)_temporal_' + clim_or_morph + '.pickle')
    out_slope_glac2.to_csv(DIR_OUT + df_out_name_base + 'slope_2var(var2)_temporal_' + clim_or_morph + '.csv')
    out_slope_glac2.to_pickle(DIR_OUT + df_out_name_base + 'slope_2var(var2)_temporal_' + clim_or_morph + '.pickle')


# ---- Martina SHORT ----#
# +++ CLIMATE +++ #
df_name_ = par_in_short
df_out_name_base_ = 'cor_spatial_MB-Martina-short_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_,temporal=True,mb_ref='Martina',period=period_short)

# ---- Martina LONG ----#
# +++ CLIMATE +++ #
df_name_ = par_in_long
df_out_name_base_ = 'cor_spatial_MB-Martina-long_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_,temporal=True,mb_ref='Martina',period=period_long)

# ---- Hugonnet SHORT ----#
# +++ CLIMATE +++ #
df_name_ = par_in_short_h
df_out_name_base_ = 'cor_spatial_MB-Hugonnet-short_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_,temporal=True,mb_ref='Hugonnet',period=period_short)

# ---- Hugonnet LONG ----#
# +++ CLIMATE +++ #
df_name_ = par_in_long_h
df_out_name_base_ = 'cor_spatial_MB-Hugonnet-long_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_,temporal=True,mb_ref='Hugonnet',period=period_long)
