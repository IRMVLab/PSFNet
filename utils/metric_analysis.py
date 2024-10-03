# -- coding: utf-8 --

import pandas as pd


def odo_metric_analysis(metric: pd.DataFrame, key_list: list) -> pd.DataFrame:
    metric_dict = {}
    for metric_name in key_list:
        for column in metric.columns:
            if metric_name in column:
                metric_dict.setdefault(metric_name, []).append(column)
            else:
                continue
    for key, value in metric_dict.items():
        df_tmp = metric[value]
        series_tmp = df_tmp.mean(axis='columns')
        metric[f'mean_{key}'] = series_tmp
        metric[f'min_{key}_epoch'] = pd.Series([series_tmp.idxmin()], index=[metric.index[0]])
