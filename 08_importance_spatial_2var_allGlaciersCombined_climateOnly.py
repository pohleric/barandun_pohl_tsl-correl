'''
SPATIAL
Calculating the relative importance as number of significant correlations per total number of tested combinations
Climate only
'''
import pandas as pd
import numpy as np
from copy import copy
from collections import Counter

mnt = r'Z:/'

# INPUT
DIR_IN = mnt + 'data/correlation_spatial_results/'
DIR_OUT = mnt + 'data/correlation_spatial_results_analysis/'

sig_lvl = 0.05
# ------------------- VARS/AGGLEVEL/STATS ------------------------- #
mb_estimates = ['Martina', 'Hugonnet']
periods = ['short', 'long']
subset_vars = ['t2', 'tas', 't2m', 'prcp', 'tp', 'pr', 'scf', 'scd']
subset_agglevels = ['hydro', 'winter', 'spring', 'summer', 'autumn']
subset_stats = ['mean', 'std', 'slope']


# ------------------- FUNCTIONS ------------------------- #
def calc_percentages(ind_ds, subset_vars, subset_agglevels, subset_stats):
    df_p_1 = pv1.iloc[:, ind_ds]
    df_p_2 = pv2.iloc[:, ind_ds]
    ind_ds_var_v1 = [i for i, v1i in enumerate(df_p_1.columns) if
                     v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels and v1i.split('_')[
                         4] in subset_stats]
    ind_ds_var_v2 = [i for i, v1i in enumerate(df_p_2.columns) if
                     v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels and v1i.split('_')[
                         4] in subset_stats]
    ind_ind_ds_var_v12 = set(ind_ds_var_v1) & set(ind_ds_var_v2)
    ind_ind_ds_var_v12 = [vi for vi in ind_ind_ds_var_v12]
    df_p_1_var = df_p_1.iloc[:, ind_ind_ds_var_v12]
    df_p_2_var = df_p_2.iloc[:, ind_ind_ds_var_v12]

    # count sign. occurrences of variables
    name_v1_sig_l = []
    name_v2_sig_l = []
    for i in range(df_p_1_var.shape[1]):
        # print(i)
        v1i = df_p_1_var.iloc[:, i]
        v2i = df_p_2_var.iloc[:, i]
        if v1i.values < sig_lvl:
            name_v1_sig_l.append(v1i.name)
        if v2i.values < sig_lvl:
            name_v2_sig_l.append(v2i.name)

    # Count the number of occurrences of a variable or season
    name_v1_sig_cnt = Counter(name_v1_sig_l)
    tab_pv1 = pd.DataFrame.from_dict(name_v1_sig_cnt, orient='index', columns=['var1'])
    name_v2_sig_cnt = Counter(name_v2_sig_l)
    tab_pv2 = pd.DataFrame.from_dict(name_v2_sig_cnt, orient='index', columns=['var2'])
    # combine
    tab_p12 = pd.merge(tab_pv1, tab_pv2, how='outer', left_index=True, right_index=True)
    tab_p12_sum = tab_p12.sum(axis=1)

    # ----- SUMMARIZING ----- #
    # =============== VARNAME =============== #
    scdf_cnt = []
    pr_cnt = []
    t2m_cnt = []
    rad_cnt = []
    wind_cnt = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn.split('_')[1]
        if tmp_vn_ == 'scf' or tmp_vn_ == 'scd':
            scdf_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'pr' or tmp_vn_ == 'prcp' or tmp_vn_ == 'tp':
            pr_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 't2' or tmp_vn_ == 'tas' or tmp_vn_ == 't2m':
            t2m_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'swdown' or tmp_vn_ == 'swup' or tmp_vn_ == 'msnswrf' or tmp_vn_ == 'msnlwrf' or tmp_vn_ == 'lwdown' or tmp_vn_ == 'lwup':
            rad_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'ws10' or tmp_vn_ == 'u10' or tmp_vn_ == 'v10':
            wind_cnt.append(tab_p12_sum[rn])

    # -------- Percentages ---------------- #
    df_t2m_percent = np.sum(t2m_cnt) / len(ind_ind_ds_var_v12)
    df_pr_percent = np.sum(pr_cnt) / len(ind_ind_ds_var_v12)
    df_scf_percent = np.sum(scdf_cnt) / len(ind_ind_ds_var_v12)
    df_rad_percent = np.sum(rad_cnt) / len(ind_ind_ds_var_v12)
    df_wind_percent = np.sum(wind_cnt) / len(ind_ind_ds_var_v12)

    # =============== STAT =============== #
    mean_cnt = []
    slope_cnt = []
    std_cnt = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn.split('_')[4]
        if tmp_vn_ == 'mean':
            mean_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'slope':
            slope_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'std':
            std_cnt.append(tab_p12_sum[rn])

    # percentage
    df_mean_percent = np.sum(mean_cnt) / len(ind_ind_ds_var_v12)
    df_slope_percent = np.sum(slope_cnt) / len(ind_ind_ds_var_v12)
    df_std_percent = np.sum(std_cnt) / len(ind_ind_ds_var_v12)

    # =============== SEASON =============== #
    hydro_cnt = []
    winter_cnt = []
    spring_cnt = []
    summer_cnt = []
    autumn_cnt = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn.split('_')[2]
        if tmp_vn_ == 'hydro':
            hydro_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'winter':
            winter_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'spring':
            spring_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'summer':
            summer_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'autumn':
            autumn_cnt.append(tab_p12_sum[rn])

    # percentage
    df_hydro_percent = np.sum(hydro_cnt) / len(ind_ind_ds_var_v12)
    df_winter_percent = np.sum(winter_cnt) / len(ind_ind_ds_var_v12)
    df_spring_percent = np.sum(spring_cnt) / len(ind_ind_ds_var_v12)
    df_summer_percent = np.sum(summer_cnt) / len(ind_ind_ds_var_v12)
    df_autumn_percent = np.sum(autumn_cnt) / len(ind_ind_ds_var_v12)

    return df_t2m_percent, df_pr_percent, df_scf_percent, df_rad_percent, df_wind_percent,\
           df_mean_percent, df_slope_percent, df_std_percent,\
           df_hydro_percent, df_winter_percent,df_spring_percent, df_summer_percent, df_autumn_percent


def calc_mean_r2(ind_ds, subset_vars, subset_agglevels, subset_stats):
    # subset using VARS/AGGLEVEL/STATS
    df_p_1 = pv1.iloc[:, ind_ds]
    df_p_2 = pv2.iloc[:, ind_ds]
    df_r_1 = rv1.iloc[:, ind_ds]
    df_r_2 = rv2.iloc[:, ind_ds]
    ind_ds_var_v1 = [i for i, v1i in enumerate(df_p_1.columns) if
                     v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels and v1i.split('_')[
                         4] in subset_stats]
    ind_ds_var_v2 = [i for i, v1i in enumerate(df_p_2.columns) if
                     v1i.split('_')[1] in subset_vars and v1i.split('_')[2] in subset_agglevels and v1i.split('_')[
                         4] in subset_stats]
    ind_ind_ds_var_v12 = set(ind_ds_var_v1) & set(ind_ds_var_v2)
    ind_ind_ds_var_v12 = [vi for vi in ind_ind_ds_var_v12]
    df_p_1_var = df_p_1.iloc[:, ind_ind_ds_var_v12]
    df_p_2_var = df_p_2.iloc[:, ind_ind_ds_var_v12]
    df_r_1_var = df_r_1.iloc[:, ind_ind_ds_var_v12]
    df_r_2_var = df_r_2.iloc[:, ind_ind_ds_var_v12]

    # count sign. occurrences of variables
    r2_v1_sig_l = []
    r2_v2_sig_l = []
    name_v1_sig_l = []
    name_v2_sig_l = []
    for i in range(df_p_1_var.shape[1]):
        # print(i)
        v1i = df_p_1_var.iloc[:, i]
        v2i = df_p_2_var.iloc[:, i]
        r1i = df_r_1_var.iloc[:, i]
        r2i = df_r_2_var.iloc[:, i]
        if v1i.values < sig_lvl:
            r2_v1_sig_l.append(r1i)
            name_v1_sig_l.append(v1i.name)
        if v2i.values < sig_lvl:
            r2_v2_sig_l.append(r2i)
            name_v2_sig_l.append(v2i.name)

    name_v1_sig_cnt = Counter(name_v1_sig_l)
    tab_pv1 = pd.DataFrame.from_dict(name_v1_sig_cnt, orient='index', columns=['var1'])
    name_v2_sig_cnt = Counter(name_v2_sig_l)
    tab_pv2 = pd.DataFrame.from_dict(name_v2_sig_cnt, orient='index', columns=['var2'])

    # p-value
    tab_p12 = pd.merge(tab_pv1,tab_pv2, how='outer', left_index=True, right_index=True)
    tab_p12_sum = tab_p12.sum(axis=1)

    # cor-coeff
    df_r_1_cat = pd.concat(r2_v1_sig_l)
    df_r_1_cat.index = name_v1_sig_l
    df_r_1_cat.index.name = 'name'
    df_r_1_cat_mean = pd.DataFrame(df_r_1_cat.groupby('name').mean())
    df_r_1_cat_mean.columns = ['r1']

    df_r_2_cat = pd.concat(r2_v2_sig_l)
    df_r_2_cat.index = name_v2_sig_l
    df_r_2_cat.index.name = 'name'
    df_r_2_cat_mean = pd.DataFrame(df_r_2_cat.groupby('name').mean())
    df_r_2_cat_mean.columns = ['r2']

    tab_r12 = pd.merge(df_r_1_cat_mean, df_r_2_cat_mean, how='outer', left_index=True, right_index=True)
    tab_r12 = tab_r12.mean(axis=1)

    # ----- SUMMARIZING ----- #
    # =============== VARNAME =============== #
    scdf_cnt = []
    pr_cnt = []
    t2m_cnt = []
    rad_cnt = []
    wind_cnt = []

    scdf_r2 = []
    pr_r2 = []
    t2m_r2 = []
    rad_r2 = []
    wind_r2 = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn.split('_')[1]
        if tmp_vn_ == 'scf' or tmp_vn_ == 'scd':
            scdf_cnt.append(tab_p12_sum[rn])
            scdf_r2.append(tab_r12[rn])
        if tmp_vn_ == 'pr' or tmp_vn_ == 'prcp' or tmp_vn_ == 'tp':
            pr_cnt.append(tab_p12_sum[rn])
            pr_r2.append(tab_r12[rn])
        if tmp_vn_ == 't2' or tmp_vn_ == 'tas' or tmp_vn_ == 't2m':
            t2m_cnt.append(tab_p12_sum[rn])
            t2m_r2.append(tab_r12[rn])
        if tmp_vn_ == 'swdown' or tmp_vn_ == 'swup' or tmp_vn_ == 'msnswrf' or tmp_vn_ == 'msnlwrf' or tmp_vn_ == 'lwdown' or tmp_vn_ == 'lwup':
            rad_cnt.append(tab_p12_sum[rn])
            rad_r2.append(tab_r12[rn])
        if tmp_vn_ == 'ws10' or tmp_vn_ == 'u10' or tmp_vn_ == 'v10':
            wind_cnt.append(tab_p12_sum[rn])
            wind_r2.append(tab_r12[rn])

    # -------- Averages ---------------- #
    df_scf_r2 = np.mean(scdf_r2)
    df_t2m_r2 = np.mean(t2m_r2)
    df_pr_r2 = np.mean(pr_r2)
    df_rad_r2 = np.mean(rad_r2)
    df_wind_r2 = np.mean(wind_r2)

    # =============== STAT =============== #
    mean_cnt = []
    slope_cnt = []
    std_cnt = []
    mean_r2 = []
    slope_r2 = []
    std_r2 = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn.split('_')[4]
        if tmp_vn_ == 'mean':
            mean_cnt.append(tab_p12_sum[rn])
            mean_r2.append(tab_r12[rn])
        if tmp_vn_ == 'slope':
            slope_cnt.append(tab_p12_sum[rn])
            slope_r2.append(tab_r12[rn])
        if tmp_vn_ == 'std':
            std_cnt.append(tab_p12_sum[rn])
            std_r2.append(tab_r12[rn])

    # -------- Averages ---------------- #
    df_mean_r2 = np.mean(mean_r2)
    df_slope_r2 = np.mean(slope_r2)
    df_std_r2 = np.mean(std_r2)

    # SEASON
    hydro_cnt = []
    winter_cnt = []
    spring_cnt = []
    summer_cnt = []
    autumn_cnt = []
    hydro_r2 = []
    winter_r2 = []
    spring_r2 = []
    summer_r2 = []
    autumn_r2 = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn.split('_')[2]
        if tmp_vn_ == 'hydro':
            hydro_cnt.append(tab_p12_sum[rn])
            hydro_r2.append(tab_r12[rn])
        if tmp_vn_ == 'winter':
            winter_cnt.append(tab_p12_sum[rn])
            winter_r2.append(tab_r12[rn])
        if tmp_vn_ == 'spring':
            spring_cnt.append(tab_p12_sum[rn])
            spring_r2.append(tab_r12[rn])
        if tmp_vn_ == 'summer':
            summer_cnt.append(tab_p12_sum[rn])
            summer_r2.append(tab_r12[rn])
        if tmp_vn_ == 'autumn':
            autumn_cnt.append(tab_p12_sum[rn])
            autumn_r2.append(tab_r12[rn])

    # -------- Averages ---------------- #
    df_hydro_r2 = np.mean(hydro_r2)
    df_winter_r2 = np.mean(winter_r2)
    df_spring_r2 = np.mean(spring_r2)
    df_summer_r2 = np.mean(summer_r2)
    df_autumn_r2 = np.mean(autumn_r2)

    return df_t2m_r2, df_pr_r2, df_scf_r2, df_rad_r2, df_wind_r2,\
           df_mean_r2, df_slope_r2, df_std_r2,\
           df_hydro_r2, df_winter_r2,df_spring_r2, df_summer_r2, df_autumn_r2


# ============================== Calculation Loop ============================== #
# using the pickle files for faster loading
for mb_estimate in mb_estimates:
    for period_length in periods:

        df_in_name_base = 'cor_spatial_MB-%s-%s_' %(mb_estimate, period_length)
        clim_or_morph = 'climateOnly'  # clim_or_morph_ = 'morphOnly'

        out_cor_glac = pd.read_pickle(DIR_IN + df_in_name_base + 'cor_2var_spatial_' + clim_or_morph + '.pickle')
        out_slope_glac = pd.read_pickle(DIR_IN + df_in_name_base + 'slope_2var_spatial_' + clim_or_morph + '.pickle')
        out_pval_glac = pd.read_pickle(DIR_IN + df_in_name_base + 'pval_2var_spatial_' + clim_or_morph + '.pickle')

        # out_pval_glac1.to_csv(DIR_OUT + df_in_name_base + 'pval_2var(var1)_spatial_' + clim_or_morph + '.csv')
        # out_pval_glac2.to_csv(DIR_OUT + df_in_name_base + 'pval_2var(var2)_spatial_' + clim_or_morph + '.csv')
        out_pval_glac1 = pd.read_pickle(DIR_IN + df_in_name_base + 'pval_2var(var1)_spatial_' + clim_or_morph + '.pickle')
        out_pval_glac2 = pd.read_pickle(DIR_IN + df_in_name_base + 'pval_2var(var2)_spatial_' + clim_or_morph + '.pickle')

        # SLOPE #
        # out_slope_glac1.to_csv(DIR_OUT + df_in_name_base + 'slope_2var(var1)_spatial_' + clim_or_morph + '.csv')
        # out_slope_glac2.to_csv(DIR_OUT + df_in_name_base + 'slope_2var(var2)_spatial_' + clim_or_morph + '.csv')
        out_slope_glac1 = pd.read_pickle(DIR_IN + df_in_name_base + 'slope_2var(var1)_spatial_' + clim_or_morph + '.pickle')
        out_slope_glac2 = pd.read_pickle(DIR_IN + df_in_name_base + 'slope_2var(var2)_spatial_' + clim_or_morph + '.pickle')

        v1 = [cni.split(';')[0] for cni in out_cor_glac.columns]
        v2 = [cni.split(';')[1] for cni in out_cor_glac.columns]

        pv1 = copy(out_pval_glac1)
        pv2 = copy(out_pval_glac2)
        rv1 = copy(out_cor_glac)
        rv2 = copy(out_cor_glac)
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
        ind_HAR30_v1 = [i for i, v1i in enumerate(v1) if 'HAR30' in v1i.split('_')[0]]
        ind_HAR30_v2 = [i for i, v1i in enumerate(v2) if 'HAR30' in v1i.split('_')[0]]
        ind_HAR30_v12 = set(ind_HAR30_v1) & set(ind_HAR30_v2)
        ind_HAR30_v12 = [vi for vi in ind_HAR30_v12]
        # ERA5
        ind_ERA5_v1 = [i for i, v1i in enumerate(v1) if 'ERA5' in v1i.split('_')[0]]
        ind_ERA5_v2 = [i for i, v1i in enumerate(v2) if 'ERA5' in v1i.split('_')[0]]
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
        ind_MHAR30_v1 = [i for i, v1i in enumerate(v1) if 'HAR30' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
        ind_MHAR30_v2 = [i for i, v1i in enumerate(v2) if 'HAR30' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
        ind_MHAR30_v12 = set(ind_MHAR30_v1) & set(ind_MHAR30_v2)
        ind_MHAR30_v12 = [vi for vi in ind_MHAR30_v12]
        # ERA5
        ind_MERA5_v1 = [i for i, v1i in enumerate(v1) if 'ERA5' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
        ind_MERA5_v2 = [i for i, v1i in enumerate(v2) if 'ERA5' in v1i.split('_')[0] or 'MOD10' in v1i.split('_')[0]]
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
        df_out_perc = pd.DataFrame(index=['t2m', 'pr', 'scf','rad','wind', 'mean','slope','std','hydro','winter','spring','summer','autumn'])
        df_out_r2 = pd.DataFrame(index=['t2m', 'pr', 'scf','rad','wind', 'mean','slope','std','hydro','winter','spring','summer','autumn'])
        for keys in dataset_combs_ind:
            print(keys)
            # skip if entry is empty:
            # - HAR e.g. has no values for long-period!
            if len(dataset_combs_ind[keys]) == 0:
                continue
            t2m, pr, scf, rad, wind, mean, slope, std,hydro, winter, spring, summer, autumn = calc_percentages(dataset_combs_ind[keys],subset_vars,subset_agglevels,subset_stats)
            df_out_perc[keys] = [t2m, pr, scf, rad, wind, mean, slope, std,hydro, winter, spring, summer, autumn]
            t2m, pr, scf, rad, wind, mean, slope, std,hydro, winter, spring, summer, autumn = calc_mean_r2(dataset_combs_ind[keys],subset_vars,subset_agglevels,subset_stats)
            df_out_r2[keys] = [t2m, pr, scf, rad, wind, mean, slope, std,hydro, winter, spring, summer, autumn]

        df_out_perc.to_csv(DIR_OUT+'percentage_'+mb_estimate+'_'+period_length+'_percentage_all_regions.csv')
        df_out_r2.to_csv(DIR_OUT+'r2_'+mb_estimate+'_'+period_length+'_percentage_all_regions.csv')


