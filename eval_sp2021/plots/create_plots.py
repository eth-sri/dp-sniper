from typing import Dict

import pandas as pd
import glob
import os
import json
import numpy as np
import sri_plot_helper as sph
import matplotlib.ticker as ticker
import math
import argparse

figure_height = 20
x_axis_label_offset_top = -0.027

# color configuration
color_reg = '#183646'
color_mlp = '#115c8e'
color_statdp_1 = '#b784a8'
color_statdp_2 = '#d7bbc7'


class DataReader:
    """
    A helper class for reading log data.
    """

    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir

    def read_data(self, experiment_label: str) -> Dict:
        """
        Read the data for a given label.
        """
        # keys: data_type, values: information for this data_type
        data = {}

        # filename pattern to cover
        pattern = os.path.join(self.logs_dir, experiment_label + "_data.log")
        for filename in glob.glob(pattern):
            with open(filename, "r") as f:
                for line in f:
                    if len(line) > 1:  # skip empty lines
                        elem = json.loads(line)

                        # extract and remove context
                        mechanism = elem['ctx'][0]
                        del elem['ctx']

                        # determine data type
                        data_type = next(iter(elem))

                        if data_type not in data:
                            data[data_type] = []

                        # add context information
                        value = elem[data_type]
                        if isinstance(value, float) or isinstance(value, int):
                            row = {data_type: value}
                        else:
                            row = value
                        row['mechanism'] = mechanism

                        data[data_type].append(row)

        # convert information to data frame
        for data_type in data.keys():
            df = pd.DataFrame(data[data_type])

            if 'mechanism' in df.columns:
                # improve naming
                rename = {
                    **{f'SparseVectorTechnique{i}': f'SVT{i}' for i in range(1, 7)},
                    'Rappor': 'RAPPOR',
                    'OneTimeRappor': 'OneTimeRAPPOR',
                    'TruncatedGeometricMechanism': 'TruncatedGeometric'
                }
                for old, new in rename.items():
                    df['mechanism'] = df['mechanism'].replace(old, new)

                n = df['mechanism'].value_counts().max()
                if n == 1:
                    # set index
                    df = df.set_index('mechanism')
            data[data_type] = df

        return data


def add_old_flag(df):
    """
    Mark all mechanisms which were originally evaluated in StatDP [1].

    [1] Ding, Zeyu, Yuxin Wang, Guanhong Wang, Danfeng Zhang, and Daniel Kifer.
        "Detecting Violations of Differential Privacy." In Proceedings of the 2018
        ACM SIGSAC Conference on Computer and Communications Security  - CCS ’18.
        https://doi.org/10.1145/3243734.3243818.
    """
    old_mechanisms = \
        ['NoisyHist1', 'NoisyHist2'] + \
        [f'ReportNoisyMax{i}' for i in range(1, 5)] + \
        [f'SVT{i}' for i in range(1, 7) if i != 2]

    df['old'] = False
    df.loc[old_mechanisms, 'old'] = True
    df = df.sort_values(by=['old', 'mechanism'])
    return df


def label_barh(ax, pos, val, color, to_text=lambda v: "{:.3f}".format(round(v, 3)), logindent=None, project=True):
    """
    Add horizontal bar labels.
    """
    # get axis information
    x_min = ax.get_xlim()[0]
    x_max = ax.get_xlim()[1]
    ax_width = ax.get_position().bounds[2]

    for (p, v) in zip(pos, val):
        label_indent = 0.005 * (x_max - x_min) / ax_width
        text = to_text(v)
        if text == 'nan':
            text = 'Error'
        if math.isnan(v):
            v = x_min
        if project:
            v = min(v, x_max)
            v = max(v, x_min)
        if x_min <= v <= x_max:
            if logindent:
                v *= logindent
            else:
                v += label_indent
            ax.text(v, p - 0.03, text,
                    color=color,
                    fontsize=7,
                    horizontalalignment='left',
                    verticalalignment='center'
                    )


def get_powers():
    """
    Get data for power evaluation.
    """

    # parse tool results
    reg = tool_reg_data['eps_lcb'].add_prefix('tool-reg-')
    mlp = tool_mlp_data['eps_lcb'].add_prefix('tool-mlp-')

    # parse StatDP results
    statdp_1 = statdp_1_data['statdp_result']
    statdp_1 = statdp_1[['eps_lcb', 'eps_preliminary']].add_prefix('statdp_1-')
    statdp_2 = statdp_2_data['statdp_result']
    statdp_2 = statdp_2[['eps_lcb', 'eps_preliminary']].add_prefix('statdp_2-')

    ret = reg.join(mlp, how='outer').join(statdp_1, how='outer').join(statdp_2, how='outer')
    ret = ret[['tool-reg-eps_lcb', 'tool-mlp-eps_lcb', 'statdp_1-eps_lcb', 'statdp_1-eps_preliminary',
               'statdp_2-eps_lcb', 'statdp_2-eps_preliminary']]
    ret = add_old_flag(ret)
    return ret


def plot_powers(output_dir):
    """
    Plot power evaluation.
    """

    df = get_powers()
    df_old = df[df['old']]
    df_new = df[~ df['old']]

    sph.configure_plots("IEEE", 7)

    fig, axes = sph.subplots(
        2, 1,
        figsize=(11, figure_height),
        nice_grid='x',
        gridspec_kw={'height_ratios': [len(df_old), len(df_new)]}
    )

    for df_ax, ax in zip([df_old, df_new], axes):
        mechanisms = df_ax.index.values.tolist()

        # set axis limits
        ax.set_xlim(0, 0.6)

        ind = np.arange(len(mechanisms))
        width = 0.23

        # plot data
        y = -ind
        x = df_ax['statdp_2-eps_lcb']
        ax.barh(y, df_ax['statdp_2-eps_preliminary'], width * 0.98, label='StatDP claimed (repeated)', fill=False,
                edgecolor=color_statdp_2, linewidth=0.3)
        bar_statdp_2 = ax.barh(y, x, width, label='StatDP (repeated)', color=color_statdp_2)
        label_barh(ax, y, x, color=color_statdp_2)

        y = y + width
        x = df_ax['statdp_1-eps_lcb']
        ax.barh(y, df_ax['statdp_1-eps_preliminary'], width * 0.98, label='StatDP claimed', fill=False,
                edgecolor=color_statdp_1, linewidth=0.3)
        bar_statdp_1 = ax.barh(y, x, width, label='StatDP', color=color_statdp_1)
        label_barh(ax, y, x, color=color_statdp_1)

        y = y + width
        x = df_ax['tool-mlp-eps_lcb']
        bar_tool_mlp = ax.barh(y, x, width, label='DD-Search Neural Network', color=color_mlp)
        label_barh(ax, y, x, color=color_mlp)

        y = y + width
        x = df_ax['tool-reg-eps_lcb']
        bar_tool_reg = ax.barh(y, x, width, label='DD-Search Logistic Regression', color=color_reg)
        label_barh(ax, y, x, color=color_reg)

        # label correctly
        ax.yaxis.set_ticks_position('none')
        ax.set_yticks(-ind + 1.5*width)
        ax.set_yticklabels(mechanisms)

    # set label
    axes[0].set_xlabel(r'$\xi$')
    axes[0].xaxis.set_label_coords(1.07, x_axis_label_offset_top)
    axes[1].set_xlabel(r'$\xi$')
    axes[1].xaxis.set_label_coords(1.07, x_axis_label_offset_top * len(df_old) / len(df_new))

    # fix layout
    fig.tight_layout(w_pad=0)

    # add legend (must be after fixing layout)
    axes[1].legend((bar_tool_reg, bar_tool_mlp, bar_statdp_1, bar_statdp_2),
                   ("DD-Search Logistic Regression", "DD-Search Neural Network", "StatDP-Fixed (1\\textsuperscript{st} run)", "StatDP-Fixed (2\\textsuperscript{nd} run)"), loc='upper right')

    # save output
    output_file = os.path.join(output_dir, 'eval-powers.pdf')
    sph.savefig(output_file)

    improvement_reg_1 = df['tool-reg-eps_lcb'] / df['statdp_1-eps_lcb']
    improvement_reg_2 = df['tool-reg-eps_lcb'] / df['statdp_2-eps_lcb']
    improvement_mlp_1 = df['tool-mlp-eps_lcb'] / df['statdp_1-eps_lcb']
    improvement_mlp_2 = df['tool-mlp-eps_lcb'] / df['statdp_2-eps_lcb']

    print("Max [Median] power factor (Logistic, run 1): {} [{}]".format(improvement_reg_1.max(),
                                                                        improvement_reg_1.mean()))
    print("Max [Median] power factor (Logistic, run 2): {} [{}]".format(improvement_reg_2.max(),
                                                                        improvement_reg_2.mean()))
    print("Max [Median] power factor (MLP, run 1): {} [{}]".format(improvement_mlp_1.max(),
                                                                   improvement_mlp_1.mean()))
    print("Max [Median] power factor (MLP, run 2): {} [{}]".format(improvement_mlp_2.max(),
                                                                   improvement_mlp_2.mean()))


def get_runtimes():
    reg = pd.concat([tool_reg_data['time_dd_search'], tool_reg_data['time_final_estimate_eps']], axis=1)
    reg = reg.loc[:, ~reg.columns.duplicated()]  # drop duplicate columns
    reg = reg.add_prefix('tool-reg-')

    mlp = pd.concat([tool_mlp_data['time_dd_search'], tool_mlp_data['time_final_estimate_eps']], axis=1)
    mlp = mlp.loc[:, ~mlp.columns.duplicated()]  # drop duplicate columns
    mlp = mlp.add_prefix('tool-mlp-')

    # parse StatDP results
    statdp_1 = pd.concat([statdp_1_data[k] for k in statdp_1_data.keys() if k.endswith("_time")], axis=1)\
        .add_suffix("_1")
    statdp_2 = pd.concat([statdp_2_data[k] for k in statdp_2_data.keys() if k.endswith("_time")], axis=1)\
        .add_suffix("_2")

    ret = reg.join(mlp, how='outer').join(statdp_1, how='outer').join(statdp_2, how='outer')
    ret = ret[['tool-reg-time_dd_search', 'tool-mlp-time_dd_search', 'tool-reg-time_final_estimate_eps',
               'tool-mlp-time_final_estimate_eps', 'statdp_time_1', 'statdp_time_2']]
    ret = add_old_flag(ret)
    return ret


def time_to_str(t):
    if math.isnan(t):
        return 'Error'
    seconds = t
    if seconds < 60:
        return "{:.0f}".format(round(seconds)) + "sec"
    minutes = seconds / 60
    if minutes < 60:
        return "{:.0f}".format(round(minutes)) + "min"
    hours = minutes / 60
    return "{:.0f}".format(round(hours)) + "h"


@ticker.FuncFormatter
def time_formatter(x, pos):
    return time_to_str(x)


def plot_runtimes(output_dir):
    times = get_runtimes()
    times_old = times[times['old']]
    times_new = times[~ times['old']]

    sph.configure_plots("IEEE", 7)

    fig, axes = sph.subplots(
        2, 1,
        figsize=(11, figure_height),
        nice_grid='x',
        gridspec_kw={'height_ratios': [len(times_old), len(times_new)]}
    )

    for times_ax, ax in zip([times_old, times_new], axes):
        mechanisms = times_ax.index.values.tolist()

        ax.set_xlim(1, times.max().max())

        ind = np.arange(len(mechanisms))
        width = 0.23

        y = -ind
        x = times_ax['statdp_time_2']
        ax.barh(y, x, width, label='StatDP (repeated)', color=color_statdp_2)
        label_barh(ax, y, x, color=color_statdp_2, to_text=time_to_str, logindent=1.05)

        y = y + width
        x = times_ax['statdp_time_1']
        ax.barh(y, x, width, label='StatDP', color=color_statdp_1)
        label_barh(ax, y, x, color=color_statdp_1, to_text=time_to_str, logindent=1.05)

        time_tool_mlp = times_ax['tool-mlp-time_dd_search'] + times_ax['tool-mlp-time_final_estimate_eps']
        y = y + width
        x = time_tool_mlp
        ax.barh(y, x, width, label='DD-Search MLP', color=color_mlp)
        label_barh(ax, y, x, color=color_mlp, to_text=time_to_str, logindent=1.05)

        time_tool_reg = times_ax['tool-reg-time_dd_search'] + times_ax['tool-reg-time_final_estimate_eps']
        y = y + width
        x = time_tool_reg
        ax.barh(y, x, width, label='DD-Search Logistic', color=color_reg)
        label_barh(ax, y, x, color=color_reg, to_text=time_to_str, logindent=1.05)

        # label correctly
        ax.yaxis.set_ticks_position('none')
        ax.set_yticks(-ind + 1.5*width)

        ax.set_yticklabels(mechanisms)

        # set x axis as times
        ax.set_xscale('log')
        # ax.xaxis.set_major_formatter(time_formatter)

    # set label
    axes[0].set_xlabel('sec')
    axes[0].xaxis.set_label_coords(0.98, x_axis_label_offset_top)
    axes[1].set_xlabel('sec')
    axes[1].xaxis.set_label_coords(0.98, x_axis_label_offset_top * len(times_old) / len(times_new))

    # fix layout
    fig.tight_layout()

    # save output
    output_file = os.path.join(output_dir, 'eval-runtimes.pdf')
    sph.savefig(output_file)

    speedup_reg_1 = times['statdp_time_1'] / (times['tool-reg-time_dd_search'] + times['tool-reg-time_final_estimate_eps'])
    speedup_mlp_1 = times['statdp_time_1'] / (times['tool-mlp-time_dd_search'] + times['tool-mlp-time_final_estimate_eps'])
    speedup_reg_2 = times['statdp_time_2'] / (times['tool-reg-time_dd_search'] + times['tool-reg-time_final_estimate_eps'])
    speedup_mlp_2 = times['statdp_time_2'] / (times['tool-mlp-time_dd_search'] + times['tool-mlp-time_final_estimate_eps'])

    print("Average speedup (Logistic, run 1):", speedup_reg_1.mean())
    print("Average speedup (Logistic, run 2):", speedup_reg_2.mean())
    print("Average speedup (MLP, run 1):", speedup_mlp_1.mean())
    print("Average speedup (MLP, run 2):", speedup_mlp_2.mean())


def analyze_probe_times():
    x_1 = statdp_1_data["statdp_time_one_probe"]
    x_2 = statdp_2_data["statdp_time_one_probe"]
    x = pd.concat([x_1, x_2])
    x = x.groupby('mechanism').mean()

    y = pd.concat([tool_reg_data["time_dd_search"], tool_reg_data["time_final_estimate_eps"]], axis=1)
    y["time_tool"] = y.time_dd_search + y.time_final_estimate_eps

    z = pd.concat([x, y], axis=1)
    z["speedup"] = z.statdp_time_one_probe / z.time_tool

    print(z[["time_tool", "statdp_time_one_probe", "speedup"]])
    print("Average per-probe speedup:", z["speedup"].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='the directory containing the input data logs')
    parser.add_argument('--output-dir', required=True, help='the directory to be used for the created plots')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tool_reg_data = DataReader(args.data_dir).read_data("dd_search_reg")
    tool_mlp_data = DataReader(args.data_dir).read_data("dd_search_mlp")
    statdp_1_data = DataReader(args.data_dir).read_data("statdp_1")
    statdp_2_data = DataReader(args.data_dir).read_data("statdp_2")

    plot_powers(args.output_dir)
    plot_runtimes(args.output_dir)
    analyze_probe_times()
