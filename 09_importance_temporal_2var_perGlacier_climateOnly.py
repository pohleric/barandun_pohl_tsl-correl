'''
TEMPORAL
Calculating the relative importance as number of significant correlations per total number of tested combinations
Climate only
'''

import pandas as pd
import numpy as np
from itertools import product

mnt = r'Z:/'
# REF FILE for regions:
region_classifications = ["region_code", "cluster3_mbstd", "cluster4_mbstd", "cluster5_mbstd", "cluster3_mb",
                          "cluster4_mb", "cluster5_mb",
                          'all_together']  # the last one has to be addressed with exception in the loop
ref_fname = mnt + 'data/2d_PTS_pars_static_filtered_k-means_clusters.csv'
ref_df = pd.read_csv(ref_fname, index_col=0)
# replace the "/"
ref_df['region_code'] = [stri.replace('/','-') for stri in ref_df['region_code'].values]

# INPUT
DIR_IN = mnt + 'data/correlation_temporal_results/'
DIR_OUT = mnt + 'data/correlation_temporal_results_analysis/'

sig_lvl = 0.05

# ------------------- VARS/AGGLEVEL/STATS ------------------------- #
mb_estimates = ['Martina', 'Hugonnet']
periods = ['short', 'long']
subset_vars = ['t2', 'tas', 't2m', 'prcp', 'tp', 'pr', 'scf', 'scd']
subset_agglevels = ['hydro', 'winter', 'spring', 'summer', 'autumn']
subset_stats = ['mean', 'std', 'slope']
region_cnt = 0
region_len = len(region_classifications)


# ------------------- FUNCTIONS ------------------------- #
def calc_temporal_regional(ind_ds, subset_vars, subset_agglevels):
    '''
    Calculates the regional stats for the variables and seasons.
    Not returning the stats per individual glacier
    :param ind_ds:
    :param subset_vars:
    :param subset_agglevels:
    :return: Count, Percentage (wr to all used variables), R2 (if significant)
    '''
    # subset using VARS/AGGLEVEL/STATS

    df_p_1 = pv1.iloc[:, ind_ds]
    df_p_2 = pv2.iloc[:, ind_ds]
    df_r_1 = rv1.iloc[:, ind_ds]
    df_r_2 = rv2.iloc[:, ind_ds]
    ind_ds_var_v1 = [i for i, v1i in enumerate(df_p_1.columns) if v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels]
    ind_ds_var_v2 = [i for i, v1i in enumerate(df_p_2.columns) if v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels]
    ind_ind_ds_var_v12 = set(ind_ds_var_v1) & set(ind_ds_var_v2)
    ind_ind_ds_var_v12 = [vi for vi in ind_ind_ds_var_v12]

    df_p_1_var = df_p_1.iloc[:, ind_ind_ds_var_v12]
    df_p_2_var = df_p_2.iloc[:, ind_ind_ds_var_v12]
    df_r_1_var = df_r_1.iloc[:, ind_ind_ds_var_v12]
    df_r_2_var = df_r_2.iloc[:, ind_ind_ds_var_v12]

    r2_v1_sig_l = pd.DataFrame(index=ref_ss_ind)
    r2_v2_sig_l =pd.DataFrame(index=ref_ss_ind)
    name_v1_sig_l = []
    name_v2_sig_l = []

    for i in range(df_p_1_var.shape[1]):
        # print(i)
        v1i = df_p_1_var.iloc[:, i]
        v2i = df_p_2_var.iloc[:, i]
        r1i = df_r_1_var.iloc[:, i]
        r2i = df_r_2_var.iloc[:, i]

        tmp_rl1i = r1i[v1i.values < sig_lvl]
        if tmp_rl1i.shape[0] > 0:
            r2_v1_sig_l[v1i.name] = tmp_rl1i
            name_v1_sig_l.append(v1i.name)
        tmp_rl2i = r2i[v2i.values < sig_lvl]
        if tmp_rl2i.shape[0] > 0:
            r2_v2_sig_l[v2i.name] = tmp_rl2i
            name_v2_sig_l.append(v2i.name)

    # ------ number of significant correlation for glaciers is simply the number of entries in the corr table

    # get number of significant matches per glacier ?
    # --> output standard table with glacid on left

    if r2_v1_sig_l.shape[1] > 0:
        p_v1_sig_l = r2_v1_sig_l.notna()
    else:
        p_v1_sig_l = r2_v1_sig_l
    if r2_v2_sig_l.shape[1] > 0:
        p_v2_sig_l = r2_v2_sig_l.notna()
    else:
        p_v2_sig_l = r2_v2_sig_l

    # P-Value merge together
    # p-var can be empty list or df
    if p_v1_sig_l.shape[1] > 0 and p_v2_sig_l.shape[1] > 0:
        p_concat = pd.concat((p_v1_sig_l, p_v2_sig_l))
        p_out = p_concat.groupby(p_concat.index).sum()              # .sum() --> total number of sign. correlations
    elif p_v1_sig_l.shape[1] > 0 and not p_v2_sig_l.shape[1] > 0:
        p_out = p_v1_sig_l
    elif not p_v1_sig_l.shape[1] > 0 and p_v2_sig_l.shape[1] > 0:
        p_out = p_v2_sig_l
    else:
        p_out = []

    # R2 - merge together:
    if r2_v1_sig_l.shape[1] > 0 and r2_v2_sig_l.shape[1] > 0:
        r2_concat = pd.concat((r2_v1_sig_l, r2_v2_sig_l))
        r2_out = r2_concat.groupby(r2_concat.index).mean()              # mean R2 value
    elif r2_v1_sig_l.shape[1] > 0 and not r2_v2_sig_l.shape[1] > 0:
        r2_out = r2_v1_sig_l
    elif not r2_v1_sig_l.shape[1] > 0 and r2_v2_sig_l.shape[1] > 0:
        r2_out = r2_v2_sig_l
    else:
        r2_out = []

    # ----- SUMMARIZING ----- #
    # =============== VARNAME =============== #

    var_cnt = pd.DataFrame(index=ref_ss_ind, columns=['scf','pr','t2m','rad','wind'])
    var_pct = pd.DataFrame(index=ref_ss_ind, columns=['scf','pr','t2m','rad','wind'])
    var_r2 = pd.DataFrame(index=ref_ss_ind, columns=['scf','pr','t2m','rad','wind'])

    if isinstance(r2_out, pd.DataFrame):
        ind_match = r2_out.columns.str.contains('_scd_') + r2_out.columns.str.contains('_scf_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['scf'] = r2_match.mean(axis=1)

        ind_match = p_out.columns.str.contains('_scd_') + p_out.columns.str.contains('_scf_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['scf'] = cnt_p
        var_pct['scf'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_pr_') + r2_out.columns.str.contains('_tp_') + r2_out.columns.str.contains('_prcp_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['pr'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_pr_') + p_out.columns.str.contains('_tp_')+ p_out.columns.str.contains('_prcp_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['pr'] = cnt_p
        var_pct['pr'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_t2_') + r2_out.columns.str.contains('_t2m_') + r2_out.columns.str.contains('_tas_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['t2m'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_t2_') + p_out.columns.str.contains('_t2m_')+ p_out.columns.str.contains('_tas_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['t2m'] = cnt_p
        var_pct['t2m'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_swdown_') + r2_out.columns.str.contains('_swup_') + r2_out.columns.str.contains('_msnswrf_')  + r2_out.columns.str.contains('_lwdown_')  + r2_out.columns.str.contains('_lwup_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['rad'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_swdown_') + p_out.columns.str.contains('_swup_')+ p_out.columns.str.contains('_msnswrf_')+ p_out.columns.str.contains('_lwdown_')+ p_out.columns.str.contains('_lwup_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['rad'] = cnt_p
        var_pct['rad'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_ws10_') + r2_out.columns.str.contains('_u10_') + r2_out.columns.str.contains('_v10_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['wind'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_ws10_') + p_out.columns.str.contains('_u10_')+ p_out.columns.str.contains('_v10_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['wind'] = cnt_p
        var_pct['wind'] = cnt_p / len(ind_ind_ds_var_v12)

    else:
        pass  # just keep the DF empty

    # -------- Percentages ---------------- #
    if isinstance(var_pct, pd.DataFrame):
        var_pct_out = var_pct.mean(axis=0)
        # -------- Averages ---------------- #
        var_r2_out = var_r2.mean(axis=0)
        # -------- Count ---------------- #
        var_cnt_out = var_cnt.sum(axis=0)
    else:
        var_pct_out = []
        # -------- Averages ---------------- #
        var_r2_out = []
        # -------- Count ---------------- #
        var_cnt_out = []

    # =============== STAT =============== #
    season_r2 = pd.DataFrame(index=ref_ss_ind, columns=['hydro','winter','spring','summer','autumn'])
    season_cnt = pd.DataFrame(index=ref_ss_ind, columns=['hydro', 'winter', 'spring', 'summer', 'autumn'])
    season_pct = pd.DataFrame(index=ref_ss_ind, columns=['hydro', 'winter', 'spring', 'summer', 'autumn'])

    if isinstance(r2_out, pd.DataFrame):
        ind_match = r2_out.columns.str.contains('_hydro')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['hydro'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_hydro')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['hydro'] = cnt_p
        season_pct['hydro'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_winter')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['winter'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_winter')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['winter'] = cnt_p
        season_pct['winter'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_spring')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['spring'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_spring')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['spring'] = cnt_p
        season_pct['spring'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_summer')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['summer'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_summer')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['summer'] = cnt_p
        season_pct['summer'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_autumn')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['autumn'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_autumn')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['autumn'] = cnt_p
        season_pct['autumn'] = cnt_p / len(ind_ind_ds_var_v12)

        # -------- Percentages ---------------- #
    if isinstance(var_pct, pd.DataFrame):
        season_pct_out = season_pct.mean(axis=0)
        # -------- Averages ---------------- #
        season_r2_out = season_r2.mean(axis=0)
        # -------- Count ---------------- #
        season_cnt_out = season_cnt.sum(axis=0)
    else:
        season_pct_out = []
        # -------- Averages ---------------- #
        season_r2_out = []
        # -------- Count ---------------- #
        season_cnt_out = []

    return var_pct_out, var_r2_out, var_cnt_out,season_pct_out, season_r2_out, season_cnt_out


def calc_temporal_perGlac(ind_ds, subset_vars, subset_agglevels):
    '''
    Calculates the glacier stats for the variables and seasons.
    :param ind_ds:
    :param subset_vars:
    :param subset_agglevels:
    :return: Count, Percentage (wr to all used variables), R2 (if significant)
    '''
    # subset using VARS/AGGLEVEL/STATS
    df_p_1 = pv1.iloc[:, ind_ds]
    df_p_2 = pv2.iloc[:, ind_ds]
    df_r_1 = rv1.iloc[:, ind_ds]
    df_r_2 = rv2.iloc[:, ind_ds]
    ind_ds_var_v1 = [i for i, v1i in enumerate(df_p_1.columns) if v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels]
    ind_ds_var_v2 = [i for i, v1i in enumerate(df_p_2.columns) if v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels]
    ind_ind_ds_var_v12 = set(ind_ds_var_v1) & set(ind_ds_var_v2)
    ind_ind_ds_var_v12 = [vi for vi in ind_ind_ds_var_v12]

    df_p_1_var = df_p_1.iloc[:, ind_ind_ds_var_v12]
    df_p_2_var = df_p_2.iloc[:, ind_ind_ds_var_v12]
    df_r_1_var = df_r_1.iloc[:, ind_ind_ds_var_v12]
    df_r_2_var = df_r_2.iloc[:, ind_ind_ds_var_v12]

    r2_v1_sig_l = pd.DataFrame(index=ref_ss_ind)
    r2_v2_sig_l =pd.DataFrame(index=ref_ss_ind)
    name_v1_sig_l = []
    name_v2_sig_l = []

    for i in range(df_p_1_var.shape[1]):
        # print(i)
        v1i = df_p_1_var.iloc[:, i]
        v2i = df_p_2_var.iloc[:, i]
        r1i = df_r_1_var.iloc[:, i]
        r2i = df_r_2_var.iloc[:, i]

        tmp_rl1i = r1i[v1i.values < sig_lvl]
        if tmp_rl1i.shape[0] > 0:
            r2_v1_sig_l[v1i.name] = tmp_rl1i
            name_v1_sig_l.append(v1i.name)
        tmp_rl2i = r2i[v2i.values < sig_lvl]
        if tmp_rl2i.shape[0] > 0:
            r2_v2_sig_l[v2i.name] = tmp_rl2i
            name_v2_sig_l.append(v2i.name)

    # ------ number of significant correlation for glaciers is simply the number of entries in the corr table

    # get number of significant matches per glacier ?
    # --> output standard table with glacid on left

    if r2_v1_sig_l.shape[1] > 0:
        p_v1_sig_l = r2_v1_sig_l.notna()
    else:
        p_v1_sig_l = r2_v1_sig_l
    if r2_v2_sig_l.shape[1] > 0:
        p_v2_sig_l = r2_v2_sig_l.notna()
    else:
        p_v2_sig_l = r2_v2_sig_l

    # P-Value merge together
    # p-var can be empty list or df
    if p_v1_sig_l.shape[1] > 0 and p_v2_sig_l.shape[1] > 0:
        p_concat = pd.concat((p_v1_sig_l, p_v2_sig_l))
        p_out = p_concat.groupby(p_concat.index).sum()              # .sum() --> total number of sign. correlations
    elif p_v1_sig_l.shape[1] > 0 and not p_v2_sig_l.shape[1] > 0:
        p_out = p_v1_sig_l
    elif not p_v1_sig_l.shape[1] > 0 and p_v2_sig_l.shape[1] > 0:
        p_out = p_v2_sig_l
    else:
        p_out = []

    # R2 - merge together:
    if r2_v1_sig_l.shape[1] > 0 and r2_v2_sig_l.shape[1] > 0:
        r2_concat = pd.concat((r2_v1_sig_l, r2_v2_sig_l))
        r2_out = r2_concat.groupby(r2_concat.index).mean()              # mean R2 value
    elif r2_v1_sig_l.shape[1] > 0 and not r2_v2_sig_l.shape[1] > 0:
        r2_out = r2_v1_sig_l
    elif not r2_v1_sig_l.shape[1] > 0 and r2_v2_sig_l.shape[1] > 0:
        r2_out = r2_v2_sig_l
    else:
        r2_out = []

    # ----- SUMMARIZING ----- #
    # =============== VARNAME =============== #

    var_cnt = pd.DataFrame(index=ref_ss_ind, columns=['scf','pr','t2m','rad','wind'])
    var_pct = pd.DataFrame(index=ref_ss_ind, columns=['scf','pr','t2m','rad','wind'])
    var_r2 = pd.DataFrame(index=ref_ss_ind, columns=['scf','pr','t2m','rad','wind'])

    if isinstance(r2_out, pd.DataFrame):
        ind_match = r2_out.columns.str.contains('_scd_') + r2_out.columns.str.contains('_scf_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['scf'] = r2_match.mean(axis=1)

        ind_match = p_out.columns.str.contains('_scd_') + p_out.columns.str.contains('_scf_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['scf'] = cnt_p
        var_pct['scf'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_pr_') + r2_out.columns.str.contains('_tp_') + r2_out.columns.str.contains('_prcp_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['pr'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_pr_') + p_out.columns.str.contains('_tp_')+ p_out.columns.str.contains('_prcp_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['pr'] = cnt_p
        var_pct['pr'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_t2_') + r2_out.columns.str.contains('_t2m_') + r2_out.columns.str.contains('_tas_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['t2m'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_t2_') + p_out.columns.str.contains('_t2m_')+ p_out.columns.str.contains('_tas_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['t2m'] = cnt_p
        var_pct['t2m'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_swdown_') + r2_out.columns.str.contains('_swup_') + r2_out.columns.str.contains('_msnswrf_')  + r2_out.columns.str.contains('_lwdown_')  + r2_out.columns.str.contains('_lwup_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['rad'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_swdown_') + p_out.columns.str.contains('_swup_')+ p_out.columns.str.contains('_msnswrf_')+ p_out.columns.str.contains('_lwdown_')+ p_out.columns.str.contains('_lwup_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['rad'] = cnt_p
        var_pct['rad'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_ws10_') + r2_out.columns.str.contains('_u10_') + r2_out.columns.str.contains('_v10_')
        r2_match = r2_out.loc[:,ind_match]
        var_r2['wind'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_ws10_') + p_out.columns.str.contains('_u10_')+ p_out.columns.str.contains('_v10_')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        var_cnt['wind'] = cnt_p
        var_pct['wind'] = cnt_p / len(ind_ind_ds_var_v12)

    else:
        pass  # just keep the DF empty

    # -------- Percentages ---------------- #
    if isinstance(var_pct, pd.DataFrame):
        var_pct_out = var_pct.mean(axis=0)
        # -------- Averages ---------------- #
        var_r2_out = var_r2.mean(axis=0)
        # -------- Count ---------------- #
        var_cnt_out = var_cnt.sum(axis=0)
    else:
        var_pct_out = []
        # -------- Averages ---------------- #
        var_r2_out = []
        # -------- Count ---------------- #
        var_cnt_out = []

    # =============== STAT =============== #
    season_r2 = pd.DataFrame(index=ref_ss_ind, columns=['hydro','winter','spring','summer','autumn'])
    season_cnt = pd.DataFrame(index=ref_ss_ind, columns=['hydro', 'winter', 'spring', 'summer', 'autumn'])
    season_pct = pd.DataFrame(index=ref_ss_ind, columns=['hydro', 'winter', 'spring', 'summer', 'autumn'])

    if isinstance(r2_out, pd.DataFrame):
        ind_match = r2_out.columns.str.contains('_hydro')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['hydro'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_hydro')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['hydro'] = cnt_p
        season_pct['hydro'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_winter')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['winter'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_winter')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['winter'] = cnt_p
        season_pct['winter'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_spring')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['spring'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_spring')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['spring'] = cnt_p
        season_pct['spring'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_summer')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['summer'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_summer')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['summer'] = cnt_p
        season_pct['summer'] = cnt_p / len(ind_ind_ds_var_v12)

        ind_match = r2_out.columns.str.contains('_autumn')
        r2_match = r2_out.loc[:, ind_match]
        season_r2['autumn'] = r2_match.mean(axis=1)
        ind_match = p_out.columns.str.contains('_autumn')
        p_match = p_out.loc[:, ind_match]
        cnt_p = p_match.sum(axis=1)
        season_cnt['autumn'] = cnt_p
        season_pct['autumn'] = cnt_p / len(ind_ind_ds_var_v12)

        # -------- Percentages ---------------- #
    if isinstance(var_pct, pd.DataFrame):
        season_pct_out = season_pct.mean(axis=0)
        # -------- Averages ---------------- #
        season_r2_out = season_r2.mean(axis=0)
        # -------- Count ---------------- #
        season_cnt_out = season_cnt.sum(axis=0)
    else:
        season_pct_out = []
        # -------- Averages ---------------- #
        season_r2_out = []
        # -------- Count ---------------- #
        season_cnt_out = []

    return var_pct, var_r2, var_cnt,season_pct, season_r2, season_cnt


# ============================== Calculation Loop ============================== #
# using the pickle files for faster loading
# Write everything into these output files
all_out_prc_pr = pd.DataFrame()
all_out_cnt_pr = pd.DataFrame()
all_out_r2_pr = pd.DataFrame()
all_out_prc_pg = pd.DataFrame(index=ref_df.index)
all_out_cnt_pg = pd.DataFrame(index=ref_df.index)
all_out_r2_pg = pd.DataFrame(index=ref_df.index)

for region_class in region_classifications:
    region_cnt += 1
    print('Working on %s of %s'%(region_cnt,region_len) )
    if not region_class == 'all_together':
        regions = ref_df[region_class]
        regions_uq = np.unique(regions.values)
    else:
        regions = ['all_together']
        regions_uq = ['alltogether']

    for region_i in regions_uq:
        if not region_i == 'alltogether':
            ref_ss = ref_df.loc[ref_df[region_class] == region_i,region_class]
            ref_ss_ind = ref_ss.index
        else:
            ref_ss = ref_df
            ref_ss_ind = ref_ss.index

        for mb_estimate in mb_estimates:
            for period_length in periods:

                df_in_name_base = 'cor_spatial_MB-%s-%s_' %(mb_estimate, period_length)
                clim_or_morph = 'climateOnly'  # clim_or_morph_ = 'morphOnly'

                out_cor_glac = pd.read_pickle(DIR_IN + df_in_name_base + 'cor_2var_temporal_' + clim_or_morph + '.pickle')
                out_cor_glac = out_cor_glac.loc[ref_ss_ind]
                out_slope_glac = pd.read_pickle(DIR_IN + df_in_name_base + 'slope_2var_temporal_' + clim_or_morph + '.pickle')
                out_slope_glac = out_slope_glac.loc[ref_ss_ind]
                out_pval_glac = pd.read_pickle(DIR_IN + df_in_name_base + 'pval_2var_temporal_' + clim_or_morph + '.pickle')
                out_pval_glac = out_pval_glac.loc[ref_ss_ind]

                # out_pval_glac1.to_csv(DIR_OUT + df_in_name_base + 'pval_2var(var1)_temporal_' + clim_or_morph + '.csv')
                # out_pval_glac2.to_csv(DIR_OUT + df_in_name_base + 'pval_2var(var2)_temporal_' + clim_or_morph + '.csv')
                out_pval_glac1 = pd.read_pickle(DIR_IN + df_in_name_base + 'pval_2var(var1)_temporal_' + clim_or_morph + '.pickle')
                out_pval_glac1 = out_pval_glac1.loc[ref_ss_ind]
                out_pval_glac2 = pd.read_pickle(DIR_IN + df_in_name_base + 'pval_2var(var2)_temporal_' + clim_or_morph + '.pickle')
                out_pval_glac2 = out_pval_glac2.loc[ref_ss_ind]

                # SLOPE #
                # out_slope_glac1.to_csv(DIR_OUT + df_in_name_base + 'slope_2var(var1)_temporal_' + clim_or_morph + '.csv')
                # out_slope_glac2.to_csv(DIR_OUT + df_in_name_base + 'slope_2var(var2)_temporal_' + clim_or_morph + '.csv')
                out_slope_glac1 = pd.read_pickle(DIR_IN + df_in_name_base + 'slope_2var(var1)_temporal_' + clim_or_morph + '.pickle')
                out_slope_glac1 = out_slope_glac1.loc[ref_ss_ind]
                out_slope_glac2 = pd.read_pickle(DIR_IN + df_in_name_base + 'slope_2var(var2)_temporal_' + clim_or_morph + '.pickle')
                out_slope_glac2 = out_slope_glac2.loc[ref_ss_ind]

                v1 = [cni.split(';')[0] for cni in out_cor_glac.columns]
                v2 = [cni.split(';')[1] for cni in out_cor_glac.columns]

                pv1 = out_pval_glac1
                pv2 = out_pval_glac2
                rv1 = out_cor_glac
                rv2 = out_cor_glac
                # adjust col names of COR
                rv1_names = [rcn.split(';')[0] for rcn in rv1.columns]
                rv2_names = [rcn.split(';')[1] for rcn in rv1.columns]
                rv1.columns = rv1_names
                rv2.columns = rv2_names

                ####################################################################################################
                # Make subsets:
                # ----------------- DATASET
                # - Only HAR
                # - Only ERA5
                # - Only CHELSA

                # ---- SINGLE DATASETS
                # HAR
                ind_HAR30_v1 = [i for i, v1i in enumerate(v1) if 'HAR' in v1i.split('_')[0]]
                ind_HAR30_v2 = [i for i, v1i in enumerate(v2) if 'HAR' in v1i.split('_')[0]]
                ind_HAR30_v12 = set(ind_HAR30_v1) & set(ind_HAR30_v2)
                ind_HAR30_v12 = [vi for vi in ind_HAR30_v12]
                # ERA5
                ind_ERA5_v1 = [i for i, v1i in enumerate(v1) if 'ECMWF' in v1i.split('_')[0]]
                ind_ERA5_v2 = [i for i, v1i in enumerate(v2) if 'ECMWF' in v1i.split('_')[0]]
                ind_ERA5_v12 = set(ind_ERA5_v1) & set(ind_ERA5_v2)
                ind_ERA5_v12 = [vi for vi in ind_ERA5_v12]
                # CHELSA
                ind_CHELSA_v1 = [i for i, v1i in enumerate(v1) if 'CHELSA' in v1i.split('_')[0]]
                ind_CHELSA_v2 = [i for i, v1i in enumerate(v2) if 'CHELSA' in v1i.split('_')[0]]
                ind_CHELSA_v12 = set(ind_CHELSA_v1) & set(ind_CHELSA_v2)
                ind_CHELSA_v12 = [vi for vi in ind_CHELSA_v12]
                # MOD10CM
                ind_MOD10_v1 = [i for i, v1i in enumerate(v1) if 'MOD10' in v1i.split('_')[0]]
                ind_MOD10_v2 = [i for i, v1i in enumerate(v2) if 'MOD10' in v1i.split('_')[0]]
                ind_MOD10_v12 = set(ind_MOD10_v1) & set(ind_MOD10_v2)
                ind_MOD10_v12 = [vi for vi in ind_MOD10_v12]

                # ---- SINGLE DATASETS + MOD10
                # HAR
                ind_MHAR30_v1 = [i for i, v1i in enumerate(v1) if 'HAR' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
                ind_MHAR30_v2 = [i for i, v1i in enumerate(v2) if 'HAR' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
                ind_MHAR30_v12 = set(ind_MHAR30_v1) & set(ind_MHAR30_v2)
                ind_MHAR30_v12 = [vi for vi in ind_MHAR30_v12]
                # ERA5
                ind_MERA5_v1 = [i for i, v1i in enumerate(v1) if 'ECMWF' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
                ind_MERA5_v2 = [i for i, v1i in enumerate(v2) if 'ECMWF' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
                ind_MERA5_v12 = set(ind_MERA5_v1) & set(ind_MERA5_v2)
                ind_MERA5_v12 = [vi for vi in ind_MERA5_v12]

                # CHELSA
                ind_MCHELSA_v1 = [i for i, v1i in enumerate(v1) if 'CHELSA' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
                ind_MCHELSA_v2 = [i for i, v1i in enumerate(v2) if 'CHELSA' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
                ind_MCHELSA_v12 = set(ind_MCHELSA_v1) & set(ind_MCHELSA_v2)
                ind_MCHELSA_v12 = [vi for vi in ind_MCHELSA_v12]

                # ----------------- VARIABLE SUBSETS (only the same variables to have comparable percentages)
                dataset_combs_ind = {'MCHELSA':ind_MCHELSA_v12, 'MERA5':ind_MERA5_v12, 'MHAR30':ind_MHAR30_v12,
                                     'CHELSA':ind_CHELSA_v12, 'ERA5':ind_ERA5_v12, 'HAR30':ind_HAR30_v12}

                # LOOP through different subsets
                output_combs2 = [k for k in dataset_combs_ind.keys()]
                output_combs1 = ['t2m', 'pr', 'scf','rad','wind', 'hydro','winter','spring','summer','autumn']

                output_combs = ['_'.join(li) for li in  list(product(output_combs1, output_combs2))]

                # STAT files / per glacier
                df_out_cnt_pg = pd.DataFrame( index=ref_ss_ind)  # columns=[cmb for cmb in output_combs],
                df_out_perc_pg = pd.DataFrame( index=ref_ss_ind)
                df_out_r2_pg = pd.DataFrame(  index=ref_ss_ind)

                # STAT files / per region
                df_out_cnt_pr = pd.DataFrame()  # columns=[cmb for cmb in output_combs]
                df_out_perc_pr = pd.DataFrame()  #(columns=[cmb for cmb in output_combs])
                df_out_r2_pr = pd.DataFrame()  #(columns=[cmb for cmb in output_combs])

                for keys in dataset_combs_ind:
                    print(keys)
                    # keys = 'MCHELSA'
                    # skip if entry is empty:
                    # - HAR e.g. has no values for long-period!
                    if len(dataset_combs_ind[keys]) == 0:
                        continue

                    var_pct_out, var_r2_out, var_cnt_out, season_pct_out, season_r2_out, season_cnt_out = calc_temporal_regional(dataset_combs_ind[keys],subset_vars,subset_agglevels)
                    var_pct, var_r2, var_cnt, season_pct, season_r2, season_cnt = calc_temporal_perGlac(dataset_combs_ind[keys], subset_vars, subset_agglevels)

                    # STAT files / per region
                    check_combs_var_out = list(product(var_pct_out.index.values, [keys]))
                    check_combs_season_out = list(product(season_pct_out.index.values, [keys]))
                    # STAT files / per glacier
                    check_combs_var = list(product(var_pct, [keys]))
                    check_combs_season = list(product(season_pct, [keys]))

                    tmp_cn_add = '_'+str(region_class) + '_' + str(region_i) + '_' + str(mb_estimate) + '_' + str(period_length)

                    # --- REGIONAL --- #
                    # CNT
                    # print('CNT 1')
                    tmp_var_ = pd.concat([var_cnt_out, season_cnt_out])
                    tmp_var_.index = ['_'.join(li)+tmp_cn_add for li in list(product(tmp_var_.index, [keys]))]
                    df_out_cnt_pr.loc[0, tmp_var_.index] = tmp_var_.values
                    # PCT
                    # print('CNT 2')
                    tmp_var_ = pd.concat([var_pct_out, season_pct_out])
                    tmp_var_.index = ['_'.join(li)+tmp_cn_add for li in list(product(tmp_var_.index, [keys]))]
                    df_out_perc_pr.loc[0, tmp_var_.index] = tmp_var_.values
                    # R2
                    # print('CNT 3')
                    tmp_var_ = pd.concat([var_r2_out, season_r2_out])
                    tmp_var_.index =['_'.join(li)+tmp_cn_add for li in list(product(tmp_var_.index, [keys]))]
                    df_out_r2_pr.loc[0, tmp_var_.index] = tmp_var_.values

                    # --- PER GLACIER --- #
                    # CNT
                    # print('CNT 4')
                    tmp_var_ = pd.merge(var_cnt, season_cnt, how='outer', left_index=True, right_index=True)
                    tmp_cn = ['_'.join(li) + tmp_cn_add for li in list(product(tmp_var_.columns, [keys]))]
                    tmp_var_.columns = tmp_cn
                    df_out_cnt_pg.loc[:,tmp_var_.columns] = tmp_var_
                    # PCT
                    # print('CNT 5')
                    tmp_var_ = pd.merge(var_pct, season_pct, how='outer', left_index=True, right_index=True)
                    tmp_var_.columns = ['_'.join(li)+tmp_cn_add for li in list(product(tmp_var_.columns, [keys]))]
                    df_out_perc_pg.loc[:, tmp_var_.columns] = tmp_var_
                    # R2
                    # print('CNT 6')
                    tmp_var_ = pd.merge(var_r2, season_r2, how='outer', left_index=True, right_index=True)
                    tmp_var_.columns = ['_'.join(li)+tmp_cn_add for li in list(product(tmp_var_.columns, [keys]))]
                    df_out_r2_pg.loc[:, tmp_var_.columns] = tmp_var_

                # Add into the predefined matrices
                all_out_prc_pr[df_out_perc_pr.columns] = df_out_perc_pr
                all_out_prc_pg[df_out_perc_pg.columns] = df_out_perc_pg
                all_out_cnt_pr[df_out_cnt_pr.columns] = df_out_cnt_pr
                all_out_cnt_pg[df_out_cnt_pg.columns] = df_out_cnt_pg
                all_out_r2_pr[df_out_r2_pr.columns] = df_out_r2_pr
                all_out_r2_pg[df_out_r2_pg.columns] = df_out_r2_pg

# write output
all_out_prc_pr.to_csv(DIR_OUT+'all_regions_all_classes_percentage_per_regions.csv')
all_out_prc_pg.to_csv(DIR_OUT+'all_regions_all_classes_percentage_per_glacier.csv')
all_out_cnt_pr.to_csv(DIR_OUT+'all_regions_all_classes_counts_per_regions.csv')
all_out_cnt_pg.to_csv(DIR_OUT+'all_regions_all_classes_counts_per_glacier.csv')
all_out_r2_pr.to_csv(DIR_OUT+'all_regions_all_classes_r2_per_regions.csv')
all_out_r2_pg.to_csv(DIR_OUT+'all_regions_all_classes_r2_per_glacier.csv')
