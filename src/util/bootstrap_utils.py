import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def compute_boostrap(df_data, v_column_names, label_name, filepath, percentage_resampling=0.8, repetitions=1000, flag_save_importance=False):

    list_variable_importance = []

    for i in range(len(v_column_names)):
        list_variable_importance.append(bootstrap_ci(df_data, v_column_names[i], (1 / percentage_resampling),
                                                     label_name, repetitions=repetitions))

    m_variable_importance = np.array(list_variable_importance)
    m_variable_importance = m_variable_importance.reshape(m_variable_importance.shape[0], 2)

    m_var_importance_bootstrap = np.c_[v_column_names, m_variable_importance]
    df_var_importance_bootstrap = pd.DataFrame(m_var_importance_bootstrap, columns=['var_name', 'ci_lower', 'ci_upper'])

    if flag_save_importance:
        df_var_importance_bootstrap.to_csv(filepath)
    
    return df_var_importance_bootstrap


def bootstrap_ci(df_data, variable, div, classes, repetitions=3000, alpha=0.05, random_state=np.arange(1, 30000, 10)):
    """
    Compute bootstrap method with confidence interval
    :param df_data: a dataframe that includes observations of the two sample (df models)
    :param variable: a column_name of the column that includes observations (sens, espec,acc,precision,auc)
    :param div:
    :param classes: a column_name of the column that includes classes (This column should contain two different group names: dt and lasso, dt and svm, svm and lasso)
    :param repetitions: the number of times you want the bootstrapping to repeat. Default is 1000.
    :param alpha: likelihood that the true population parameter lies outside the confidence interval. Default is 0.05.
    :param random_state: enable users to set their own random_state, default is None
    :return:
    """
    df = df_data[[variable, classes]]
    bootstrap_sample_size = len(df)

    proportion_diffs = []
    print("sample size: ", int(bootstrap_sample_size/div))

    for i in range(repetitions):

        bootstrap_sample = df.sample(n=int(bootstrap_sample_size/div), replace=False, random_state=random_state[i])

        print(bootstrap_sample)
        if variable=='streaks' or variable=='color':

            label_0 = (bootstrap_sample[(bootstrap_sample[classes] == 0)])
            mean_label_0 =label_0[variable].mean()
            label_1 = (bootstrap_sample[(bootstrap_sample[classes] == 1)])
            mean_label_1 = label_1[variable].mean()

            proportion_diff = mean_label_1 - mean_label_0
            proportion_diffs.append(proportion_diff)

            df_proportions = pd.DataFrame(proportion_diffs)
        else:
            label_0 = (bootstrap_sample[(bootstrap_sample[classes] == 0)])
            mean_label_0 = np.median(label_0[variable])
            label_1 = (bootstrap_sample[(bootstrap_sample[classes] == 1)])
            mean_label_1 = np.median(label_1[variable])

            proportion_diff = mean_label_1 - mean_label_0
            proportion_diffs.append(proportion_diff)

            df_proportions = pd.DataFrame(proportion_diffs)

    out = []

    for tt in range(len(df_proportions.columns)):

        left = np.percentile(df_proportions[tt], alpha / 2 * 100)
        right = np.percentile(df_proportions[tt], 100 - alpha / 2 * 100)
        out.append([round(left, 2), round(right, 2)])

    return out


