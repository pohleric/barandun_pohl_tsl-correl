'''
SPATIAL
Calculating the relative importance as number of significant correlations per total number of tested combinations
Morphology only
'''
import pandas as pd
import numpy as np
from copy import copy
from collections import Counter

mnt = r'Z:/'

# INPUT
DIR_IN = mnt + 'data/correlation_spatial_results/'
DIR_OUT = mnt + 'data/correlation_spatial_results_analysis_morphOnly/'

sig_lvl = 0.05
# ------------------- VARS/AGGLEVEL/STATS ------------------------- #
mb_estimates = ['Martina', 'Hugonnet']
periods = ['short', 'long']
subset_morph = ['lat ', 'long', 'med. elev', 'aspect', 'slope over tongue', 'area ', 'asp_cos', 'asp_sin', 'carea']


# ------------------- FUNCTIONS ------------------------- #
def calc_percentages_morph(subset_morph):
    # subset using VARS/AGGLEVEL/STATS
    df_p_1 = pv1  # .iloc  #[:, ind_ds]
    df_p_2 = pv2  # .iloc  #[:, ind_ds]
    ind_ds_var_v1 = [i for i, v1i in enumerate(df_p_1.columns) if v1i in subset_morph]
    ind_ds_var_v2 = [i for i, v1i in enumerate(df_p_2.columns) if v1i in subset_morph]
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

    name_v1_sig_cnt = Counter(name_v1_sig_l)
    tab_pv1 = pd.DataFrame.from_dict(name_v1_sig_cnt, orient='index', columns=['var1'])
    name_v2_sig_cnt = Counter(name_v2_sig_l)
    tab_pv2 = pd.DataFrame.from_dict(name_v2_sig_cnt, orient='index', columns=['var2'])

    tab_p12 = pd.merge(tab_pv1, tab_pv2, how='outer', left_index=True, right_index=True)
    tab_p12_sum = tab_p12.sum(axis=1)

    # ----- SUMMARIZING ----- #
    # =============== VARNAME =============== #
    elev_cnt = []
    aspect_cnt = []
    slope_cnt = []
    lat_cnt = []
    lon_cnt = []
    area_cnt = []
    carea_cnt = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn
        if tmp_vn_ == 'med. elev':
            elev_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'aspect' or tmp_vn_ == 'asp_cos' or tmp_vn_ == 'asp_sin':
            aspect_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'slope over tongue':
            slope_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'lat ':
            lat_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'long':
            lon_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'area ':
            area_cnt.append(tab_p12_sum[rn])
        if tmp_vn_ == 'carea ':
            carea_cnt.append(tab_p12_sum[rn])

    # -------- Percentages ---------------- #
    df_elev_percent = np.sum(elev_cnt) / len(ind_ind_ds_var_v12)
    df_aspect_percent = np.sum(aspect_cnt) / len(ind_ind_ds_var_v12)
    df_slope_percent = np.sum(slope_cnt) / len(ind_ind_ds_var_v12)
    df_lat_percent = np.sum(lat_cnt) / len(ind_ind_ds_var_v12)
    df_lon_percent = np.sum(lon_cnt) / len(ind_ind_ds_var_v12)
    df_area_percent = np.sum(area_cnt) / len(ind_ind_ds_var_v12)
    df_carea_percent = np.sum(carea_cnt) / len(ind_ind_ds_var_v12)

    return df_elev_percent, df_aspect_percent, df_slope_percent, df_lat_percent, df_lon_percent, df_area_percent,  df_carea_percent


def calc_mean_r2_morph(subset_morph):
    # subset using VARS/AGGLEVEL/STATS
    df_p_1 = pv1  # .iloc  #[:, ind_ds]  # not necessary
    df_p_2 = pv2  # .iloc  #[:, ind_ds]
    df_r_1 = rv1  # .iloc[:, ind_ds]
    df_r_2 = rv2  # .iloc[:, ind_ds]
    ind_ds_var_v1 = [i for i, v1i in enumerate(df_p_1.columns) if v1i in subset_morph]
    ind_ds_var_v2 = [i for i, v1i in enumerate(df_p_2.columns) if v1i in subset_morph]
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
    tab_p12 = pd.merge(tab_pv1, tab_pv2, how='outer', left_index=True, right_index=True)
    tab_p12_sum = tab_p12.sum(axis=1)

    # cor-coeff
    if not len(r2_v1_sig_l) == 0:
        df_r_1_cat = pd.concat(r2_v1_sig_l)
        df_r_1_cat.index = name_v1_sig_l
    else:
        df_r_1_cat = pd.DataFrame([0])
        df_r_1_cat.index = [0]
    df_r_1_cat.index.name = 'name'
    df_r_1_cat_mean = pd.DataFrame(df_r_1_cat.groupby('name').mean())
    df_r_1_cat_mean.columns = ['r1']

    if not len(r2_v2_sig_l) == 0:
        df_r_2_cat = pd.concat(r2_v2_sig_l)
        df_r_2_cat.index = name_v2_sig_l
    else:
        df_r_2_cat = pd.DataFrame([0])
        df_r_2_cat.index = [0]
    df_r_2_cat.index.name = 'name'
    df_r_2_cat_mean = pd.DataFrame(df_r_2_cat.groupby('name').mean())
    df_r_2_cat_mean.columns = ['r2']

    tab_r12 = pd.merge(df_r_1_cat_mean, df_r_2_cat_mean, how='outer', left_index=True, right_index=True)
    tab_r12 = tab_r12.mean(axis=1)

    # ----- SUMMARIZING ----- #
    # =============== VARNAME =============== #
    elev_cnt = []
    aspect_cnt = []
    slope_cnt = []
    lat_cnt = []
    lon_cnt = []
    area_cnt = []
    carea_cnt = []

    elev_r2 = []
    aspect_r2 = []
    slope_r2 = []
    lat_r2 = []
    lon_r2 = []
    area_r2 = []
    carea_r2 = []
    for rn in tab_p12_sum.index:
        tmp_vn_ = rn
        if tmp_vn_ == 'med. elev':
            elev_cnt.append(tab_p12_sum[rn])
            elev_r2.append(tab_r12[rn])

        if tmp_vn_ == 'aspect' or tmp_vn_ == 'asp_cos' or tmp_vn_ == 'asp_sin':
            aspect_cnt.append(tab_p12_sum[rn])
            aspect_r2.append(tab_r12[rn])

        if tmp_vn_ == 'slope over tongue':
            slope_cnt.append(tab_p12_sum[rn])
            slope_r2.append(tab_r12[rn])

        if tmp_vn_ == 'lat ':
            lat_cnt.append(tab_p12_sum[rn])
            lat_r2.append(tab_r12[rn])

        if tmp_vn_ == 'long':
            lon_cnt.append(tab_p12_sum[rn])
            lon_r2.append(tab_r12[rn])

        if tmp_vn_ == 'area ':
            area_cnt.append(tab_p12_sum[rn])
            area_r2.append(tab_r12[rn])

        if tmp_vn_ == 'carea':
            carea_cnt.append(tab_p12_sum[rn])
            carea_r2.append(tab_r12[rn])

    # -------- Averages ---------------- #
    df_elev_r2 = np.mean(elev_r2)
    df_aspect_r2 = np.mean(aspect_r2)
    df_slope_r2 = np.mean(slope_r2)
    df_lat_r2 = np.mean(lat_r2)
    df_lon_r2 = np.mean(lon_r2)
    df_area_r2 = np.mean(area_r2)
    df_carea_r2 = np.mean(carea_r2)

    return df_elev_r2, df_aspect_r2, df_slope_r2, df_lat_r2, df_lon_r2, df_area_r2,  df_carea_r2


# ============================== Calculation Loop ============================== #
# using the pickle files for faster loading
for mb_estimate in mb_estimates:
    for period_length in periods:

        df_in_name_base = 'cor_spatial_MB-%s-%s_' %(mb_estimate, period_length)
        clim_or_morph = 'morphOnly'  # clim_or_morph_ = 'morphOnly'

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
        # - MORPH ONLY
        # LOOP through different subsets
        dataset_combs_ind = {'MORPH': out_pval_glac.columns}
        df_out_perc = pd.DataFrame(index=['lat ',
                                          'long',
                                          'med. elev',
                                          'aspect',
                                          'slope over tongue',
                                          'area ',
                                          'carea'])
        df_out_r2 = pd.DataFrame(index=['lat ',
                                        'long',
                                        'med. elev',
                                        'aspect',
                                        'slope over tongue',
                                        'area ',
                                        'carea'])

        for keys in dataset_combs_ind:
            print(keys)
            # skip if entry is empty:
            # - HAR e.g. has no values for long-period!
            if len(dataset_combs_ind[keys]) == 0:
                continue
            elev_percent, aspect_percent, slope_percent, lat_percent, lon_percent, area_percent, carea_percent = calc_percentages_morph(subset_morph)
            df_out_perc[keys] = [elev_percent, aspect_percent, slope_percent, lat_percent, lon_percent,
                                 area_percent, carea_percent]

            df_elev_r2, df_aspect_r2, df_slope_r2, df_lat_r2, df_lon_r2, df_area_r2, df_carea_r2 = calc_mean_r2_morph( subset_morph)
            df_out_r2[keys] = [df_elev_r2, df_aspect_r2, df_slope_r2, df_lat_r2, df_lon_r2, df_area_r2, df_carea_r2]

        df_out_perc.to_csv(DIR_OUT+'percentage_'+mb_estimate+'_'+period_length+'_percentage_all_regions.csv')
        df_out_r2.to_csv(DIR_OUT+'r2_'+mb_estimate+'_'+period_length+'_percentage_all_regions.csv')


