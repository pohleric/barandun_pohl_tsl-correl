"""
SPATIAL
Correlation analysis - using always a combination of two regressors.
All glaciers combined
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

# INPUT PAR files for spatial correlation in R - Estimates MARTINA
par_input_outfile_short = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short.csv'
par_input_outfile_short_morph = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short_morph.csv'
par_input_outfile_long = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long.csv'
par_input_outfile_long_morph = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long_morph.csv'

# INPUT PAR files for spatial correlation in R - Estimates HUGONNET
par_input_outfile_short_h = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short_hugonnet.csv'
par_input_outfile_short_morph_h = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_short_morph_hugonnet.csv'
par_input_outfile_long_h = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long_hugonnet.csv'
par_input_outfile_long_morph_h = mnt + 'data/correlation_spatial/2d_PTS_filtered_corInput_spatial_long_morph_hugonnet.csv'

DIR_OUT = mnt + 'data/correlation_spatial_results/'

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
par_morph = [
    'med. elev',
    'aspect',
    'slope over tongue',
    'area ',  # with spacebar
    'asp_cos',
    'asp_sin',
    'lat ',  # with spacebar
    'long',
    'carea'
]

aggregation_levels = ['hydro', 'winter', 'spring', 'summer', 'autumn']
dataset_names = ['ERA5', 'HAR30', 'CHELSA', 'MOD10']


# --------- Spatial correlation --------- #
# output arrays n*m ,n=glaciers, m=variables
# - p-value
# - R2

# ============================== VARIABLES ============================== #


def compute_and_write(df_name, df_out_name_base, clim_or_morph, var_to_use):

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
    df_dep = df['mb']
    vars_ind = df_ind.columns.values

    # Corr mat each glacier - all variable combinations
    comb0 = []
    a = combinations(vars_ind, 2)
    for t in a:
        comb0.append(t)
    if clim_or_morph == 'climateOnly':
        combs = [ci[0] + ';' + ci[1] for ci in comb0 if ci[0] != ci[1] and (ci[0].split('_')[0] == ci[1].split('_')[0] or ci[0].split('_')[0] =='MOD10' or  ci[1].split('_')[0] =='MOD10')]
    else:
        combs = [ci[0] + ';' + ci[1] for ci in comb0 if ci[0] != ci[1] ]
    len(combs)
    # # only combinations that consist of two different variables
    # combinations = [ci[0]+';'+ci[1] for ci in comb0 if (ci[0].split('_')[0] != ci[1].split('_')[0] )]
    # len(combinations)

    out_cor_glac = pd.DataFrame(columns=combs, index=[0])
    out_pval_glac = copy(out_cor_glac)
    out_slope_glac = copy(out_cor_glac)

    vr_ix1 = {}
    vr_ix2 = {}
    for c0, c1 in comb0:
        ind1 = [c0 == vars_ind_i for vars_ind_i in vars_ind]
        ind2 = [c1 == vars_ind_i for vars_ind_i in vars_ind]
        ind_comb1 = ind1
        ind_comb2 = ind2
        tmp_str1 = c0
        tmp_str2 = c1
        vr_ix1[tmp_str1] = ind_comb1
        vr_ix2[tmp_str2] = ind_comb2

    # Takes around 1h:40m on a single core to compute (lx) !!!!
    vars_ind_unq_2var = combs
    y = df_dep.to_numpy().reshape((-1, 1))

    for vars_i in tqdm(vars_ind_unq_2var):
        vr1, vr2 = vars_i.split(';')

        ind1 = np.array(vr_ix1[vr1])
        ind2 = np.array(vr_ix2[vr2])

        x1 = df_ind.iloc[:, ind1].values
        x2 = df_ind.iloc[:, ind2].values
        # if two many NAs for correlation, just skip
        if np.isnan(x2).sum() > 1000:
            continue
        if np.isnan(x1).sum() > 1000:
            continue
        # remove Nan
        nan_ix = np.isfinite([x1, x2, y]).all(axis=0)

        xs = np.hstack([x1, x2])[nan_ix[:, 0], :]
        Y = y[nan_ix].reshape((-1, 1))

        # strip empty spaces for LR
        vr1 = vr1.replace(' ', '_')
        vr2 = vr2.replace(' ', '_')
        # strip "-" symbol
        vr1 = vr1.replace('-', '_')
        vr2 = vr2.replace('-', '_')
        # strip "." symbol
        vr1 = vr1.replace('.', '_')
        vr2 = vr2.replace('.', '_')

        dat = pd.DataFrame(np.hstack((Y, xs)), columns=['mb', vr1, vr2])
        # corr
        fit = smf.ols('mb ~ %s + %s' % (vr1, vr2), data=dat).fit()
        slopes = fit.params[1:3].values
        slopes_str = ";".join([str(slope_i) for slope_i in slopes])
        pvalues = fit.pvalues[1:3].values
        pvalues_str = ";".join([str(pval_i) for pval_i in pvalues])
        rsquared = fit.rsquared
        if rsquared < 0:
            raise ValueError('check formula and outcome - R2 < 0')

        # write to matrix
        out_slope_glac.loc[0, vars_i] = slopes_str  # [sli for sli in slopes]
        out_cor_glac.loc[0, vars_i] = rsquared
        out_pval_glac.loc[0, vars_i] = pvalues_str

    out_cor_glac.to_csv(DIR_OUT + df_out_name_base + 'cor_2var_spatial_' + clim_or_morph + '.csv')
    out_slope_glac.to_csv(DIR_OUT + df_out_name_base + 'slope_2var_spatial_' + clim_or_morph + '.csv')
    out_pval_glac.to_csv(DIR_OUT + df_out_name_base + 'pval_2var_spatial_' + clim_or_morph + '.csv')

    out_cor_glac.to_pickle(DIR_OUT + df_out_name_base + 'cor_2var_spatial_' + clim_or_morph + '.pickle')
    out_slope_glac.to_pickle(DIR_OUT + df_out_name_base + 'slope_2var_spatial_' + clim_or_morph + '.pickle')
    out_pval_glac.to_pickle(DIR_OUT + df_out_name_base + 'pval_2var_spatial_' + clim_or_morph + '.pickle')

    # ------- split matrices into two --------- #
    # Easier to analyze individual variable significance by reading the output as individual matrices
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
    out_pval_glac1.to_csv(DIR_OUT + df_out_name_base + 'pval_2var(var1)_spatial_' + clim_or_morph + '.csv')
    out_pval_glac1.to_pickle(DIR_OUT + df_out_name_base + 'pval_2var(var1)_spatial_' + clim_or_morph + '.pickle')
    out_pval_glac2.to_csv(DIR_OUT + df_out_name_base + 'pval_2var(var2)_spatial_' + clim_or_morph + '.csv')
    out_pval_glac2.to_pickle(DIR_OUT + df_out_name_base + 'pval_2var(var2)_spatial_' + clim_or_morph + '.pickle')

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
    out_slope_glac1.to_csv(DIR_OUT + df_out_name_base + 'slope_2var(var1)_spatial_' + clim_or_morph + '.csv')
    out_slope_glac1.to_pickle(DIR_OUT + df_out_name_base + 'slope_2var(var1)_spatial_' + clim_or_morph + '.pickle')
    out_slope_glac2.to_csv(DIR_OUT + df_out_name_base + 'slope_2var(var2)_spatial_' + clim_or_morph + '.csv')
    out_slope_glac2.to_pickle(DIR_OUT + df_out_name_base + 'slope_2var(var2)_spatial_' + clim_or_morph + '.pickle')

    # R2 #
    out_cor_glac1 = copy(out_cor_glac)
    out_cor_glac2 = copy(out_cor_glac)

    out_cor_glac1 = out_cor_glac1.astype(float)
    out_cor_glac2 = out_cor_glac2.astype(float)
    out_cor_glac1.columns = [sti.split(';')[0] for sti in out_cor_glac1.columns]
    out_cor_glac2.columns = [sti.split(';')[1] for sti in out_cor_glac2.columns]
    out_cor_glac1.to_csv(DIR_OUT + df_out_name_base + 'cor_2var(var1)_spatial_' + clim_or_morph + '.csv')
    out_cor_glac1.to_pickle(DIR_OUT + df_out_name_base + 'cor_2var(var1)_spatial_' + clim_or_morph + '.pickle')
    out_cor_glac2.to_csv(DIR_OUT + df_out_name_base + 'cor_2var(var2)_spatial_' + clim_or_morph + '.csv')
    out_cor_glac2.to_pickle(DIR_OUT + df_out_name_base + 'slope_2var(var2)_spatial_' + clim_or_morph + '.pickle')

# ---- Martina SHORT ----#
# +++ CLIMATE +++ #
df_name_ = par_input_outfile_short
df_out_name_base_ = 'cor_spatial_MB-Martina-short_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)
# +++ MORPH +++ #
df_name_ = par_input_outfile_short_morph
df_out_name_base_ = 'cor_spatial_MB-Martina-short_'
clim_or_morph_ = 'morphOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)

# ---- Martina LONG ----#
# +++ CLIMATE +++ #
df_name_ = par_input_outfile_long
df_out_name_base_ = 'cor_spatial_MB-Martina-long_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)
# +++ MORPH +++ #
df_name_ = par_input_outfile_long_morph
df_out_name_base_ = 'cor_spatial_MB-Martina-long_'
clim_or_morph_ = 'morphOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)

# ---- Hugonnet SHORT ----#
# +++ CLIMATE +++ #
df_name_ = par_input_outfile_short_h
df_out_name_base_ = 'cor_spatial_MB-Hugonnet-short_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)
# +++ MORPH +++ #
df_name_ = par_input_outfile_short_morph_h
df_out_name_base_ = 'cor_spatial_MB-Hugonnet-short_'
clim_or_morph_ = 'morphOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)

# ---- Hugonnet LONG ----#
# +++ CLIMATE +++ #
df_name_ = par_input_outfile_long_h
df_out_name_base_ = 'cor_spatial_MB-Hugonnet-long_'
clim_or_morph_ = 'climateOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_clim  # var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)
# +++ MORPH +++ #
df_name_ = par_input_outfile_long_morph_h
df_out_name_base_ = 'cor_spatial_MB-Hugonnet-long_'
clim_or_morph_ = 'morphOnly'  # clim_or_morph_ = 'morphOnly'
var_to_use_ = par_morph
compute_and_write(df_name=df_name_, df_out_name_base=df_out_name_base_, clim_or_morph=clim_or_morph_,
                  var_to_use=var_to_use_)