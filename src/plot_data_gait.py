import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patches
import pandas as pd
import json

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from src.load_data_gait import *


def plot_segmentation_gait_events(trial):
    """Plot the final figure for step detection and save the fig in the output folder as png file.

    Parameters
    ----------
    trial {dict} -- dictionary with the trial data
        trial["metadata"] {dict} -- dictionary with the metadata from which are constructed :
            - gait_events {dictionary} -- dictionary with the detected gait events
            - seg {dict} -- dictionary with 4 segmentation limits ('gait start', 'uturn start', 'uturn end', 'gait end')
            - freq {int} -- acquisition frequency
        trial["data"] {dict} -- dictionary with pandas dataframe with raw data from the sensors

    """

    gait_events = {"LF": trial["metadata"]["leftGaitEvents"],
                   "RF": trial["metadata"]["rightGaitEvents"]}
    seg = {"gait start": min(np.min(trial["metadata"]["leftGaitEvents"]), np.min(trial["metadata"]["rightGaitEvents"])),
           "uturn start": trial["metadata"]["uturnBoundaries"][0],
           "uturn end": trial["metadata"]["uturnBoundaries"][1],
           "gait end": max(np.max(trial["metadata"]["leftGaitEvents"]), np.max(trial["metadata"]["rightGaitEvents"]))}
    freq = trial["metadata"]["freq"]

    data = trial["data_processed"]

    name = "Gait events detection - "

    fig, ax = plt.subplots(3, figsize=(20, 9), sharex=True, sharey=False, gridspec_kw={'height_ratios': [10, 1, 10]})

    ax[0].grid()
    ax[2].grid()

    # Phases segmentation
    # Phase 0: waiting
    ax[1].add_patch(patches.Rectangle((0, 0),  # (x,y)
                                      seg['gait start'] / freq,  # width
                                      1,  # height
                                      alpha=0.1, color="k"))
    ax[1].text(seg['gait start'] / (2 * freq), 0.5, 'waiting', fontsize=9, horizontalalignment='center',
               verticalalignment='center')

    # Phase 1: go
    ax[1].add_patch(patches.Rectangle((seg['gait start'] / freq, 0),  # (x,y)
                                      (seg['uturn start'] - seg['gait start']) / freq,  # width
                                      1,  # height
                                      alpha=0.2, color="k"))
    ax[1].text(seg['gait start'] / freq + (seg['uturn start'] - seg['gait start']) / (2 * freq), 0.5, 'straight (go)',
               fontsize=9,
               horizontalalignment='center', verticalalignment='center')

    # Phase 2: uturn
    ax[1].add_patch(patches.Rectangle((seg['uturn start'] / freq, 0),  # (x,y)
                                      (seg['uturn end'] - seg['uturn start']) / freq,  # width
                                      1,  # height
                                      alpha=0.3, color="k"))
    ax[1].text(seg['uturn start'] / freq + (seg['uturn end'] - seg['uturn start']) / (2 * freq), 0.5, 'uturn',
               fontsize=9,
               horizontalalignment='center', verticalalignment='center')

    # Phase 3: back
    ax[1].add_patch(patches.Rectangle((seg['uturn end'] / freq, 0),  # (x,y)
                                      (seg['gait end'] - seg['uturn end']) / freq,  # width
                                      1,  # height
                                      alpha=0.2, color="k"))
    ax[1].text(seg['uturn end'] / freq + (seg['gait end'] - seg['uturn end']) / (2 * freq), 0.5, 'straight (back)',
               fontsize=9,
               horizontalalignment='center', verticalalignment='center')

    # Phase 4: waiting
    ax[1].add_patch(patches.Rectangle((seg['gait end'] / freq, 0),  # (x,y)
                                      (len(data["PacketCounter"]) - seg['gait end']) / freq,  # width
                                      1,  # height
                                      alpha=0.1, color="k"))
    ax[1].text(seg['gait end'] / freq + (len(data["PacketCounter"]) - seg['gait end']) / (2 * freq), 0.5, 'waiting',
               fontsize=9,
               horizontalalignment='center', verticalalignment='center')

    ax[0].set(xlabel='Time (s)', ylabel='Angular velocity (rad/s)')
    ax[0].set_title(label=name + "Left Foot", weight='bold')
    ax[0].xaxis.set_tick_params(labelsize=12)
    ax[1].set(ylabel='Phases')
    ax[1].set_yticks([])
    ax[2].set(xlabel='Time (s)', ylabel='Angular velocity (rad/s)')
    ax[2].set_title(label=name + "Right Foot", weight='bold')
    ax[2].xaxis.set_tick_params(labelsize=12)

    # ----------------------- Feet -------------------------------------------
    t_sensor = data["PacketCounter"] / freq
    for sensor in ["RF", "LF"]:
        gyr_sensor = data[sensor + "_Gyr_Y"]
        if sensor == "LF":
            ax[0].plot(t_sensor, gyr_sensor)
            n_ax = 0
        else:
            ax[2].plot(t_sensor, gyr_sensor)
            n_ax = 2
        ma_sensor = max(gyr_sensor)
        mi_sensor = min(gyr_sensor)
        for i in range(len(gait_events[sensor])):
            to = int(gait_events[sensor][i][0])
            ax[n_ax].vlines(t_sensor[to], mi_sensor, ma_sensor, 'k', '--')
            hs = int(gait_events[sensor][i][1])
            ax[n_ax].vlines(t_sensor[hs], mi_sensor, ma_sensor, 'k', '--')
            ax[n_ax].add_patch(patches.Rectangle((t_sensor[to], mi_sensor),  # (x,y)
                                                 t_sensor[hs] - t_sensor[to],  # width
                                                 ma_sensor - mi_sensor,  # height
                                                 alpha=0.1,
                                                 facecolor='red', linestyle='dotted'))
            if i < len(gait_events[sensor]) - 1:
                to_ap = int(gait_events[sensor][i + 1][0])
                ax[n_ax].add_patch(patches.Rectangle((t_sensor[hs], mi_sensor),  # (x,y)
                                                     t_sensor[to_ap] - t_sensor[hs],  # width
                                                     ma_sensor - mi_sensor,  # height
                                                     alpha=0.1,
                                                     facecolor='green', linestyle='dotted'))

    # legend
    red_patch = mpatches.Patch(color='red', alpha=0.1, label='swing')
    green_patch = mpatches.Patch(color='green', alpha=0.1, label='stance')

    ax[0].legend(handles=[red_patch, green_patch], loc="upper left")
    ax[2].legend(handles=[red_patch, green_patch], loc="upper left")

    plt.show()


def plot_segmentation(trial):
    """Plot the uturn detection as a .png figure.

    Parameters
    ----------
    trial {dict} -- dictionary with the trial data
        trial["metadata"] {dict} -- dictionary with the metadata from which are extracted :
            - uturnBoundaries {list} -- ['uturn start', 'uturn end']
            - freq {int} -- acquisition frequency
        trial["data"] {dict} -- dictionary with pandas dataframe with raw data from the sensors

    """

    sensor = "LB"  # sensor to plot
    data = trial["data_processed"]
    freq = trial["metadata"]["freq"]
    seg = {"uturn start": trial["metadata"]["uturnBoundaries"][0],
           "uturn end": trial["metadata"]["uturnBoundaries"][1]}

    # data
    t = data["PacketCounter"]
    angle = (np.cumsum(data[sensor + "_Gyr_X"]) - np.cumsum(data[sensor + "_Gyr_X"])[0]) / freq
    angle = angle * 360 / (2 * np.pi)  # in degrees

    # fig initialization
    fig, ax = plt.subplots(2, figsize=(20, 8), sharex=True, sharey=False, gridspec_kw={'height_ratios': [20, 1]})
    ax[0].grid()
    ax[0].plot(t / freq, angle)
    ax[0].set_ylabel('Angular position (°)', fontsize=15)
    ax[0].set_title("uturn detection", fontsize=15, weight='bold')
    ax[0].set_xlabel('Time (s)', fontsize=15)

    # min and max
    mi = np.min(angle) - 0.05 * (np.max(angle) - np.min(angle))
    ma = np.max(angle) + 0.05 * (np.max(angle) - np.min(angle))

    # phases segmentation
    # Phase 1: go
    ax[0].add_patch(patches.Rectangle((0, mi),  # (x,y)
                                      seg['uturn start'] / freq,  # width
                                      ma - mi,  # height
                                      alpha=0.2, color="k"))
    ax[1].add_patch(patches.Rectangle((0, 0),  # (x,y)
                                      seg['uturn start'] / freq,  # width
                                      1,  # height
                                      alpha=0.2, color="k"))
    ax[1].text(seg['uturn start'] / (2 * freq), 0.5, 'straight (go)',
               fontsize=9,
               horizontalalignment='center', verticalalignment='center')

    # Phase 2: uturn
    ax[0].vlines(seg['uturn start'] / freq, mi, ma, 'black', '--', linewidth=3, label="uturn boundaries")
    ax[0].add_patch(patches.Rectangle((seg['uturn start'] / freq, mi),  # (x,y)
                                      (seg['uturn end'] - seg['uturn start']) / freq,  # width
                                      ma - mi,  # height
                                      alpha=0.3, color="k"))
    ax[1].add_patch(patches.Rectangle((seg['uturn start'] / freq, 0),  # (x,y)
                                      (seg['uturn end'] - seg['uturn start']) / freq,  # width
                                      1,  # height
                                      alpha=0.3, color="k"))
    ax[1].text(seg['uturn start'] / freq + (seg['uturn end'] - seg['uturn start']) / (2 * freq), 0.5, 'uturn',
               fontsize=9,
               horizontalalignment='center', verticalalignment='center')

    # Phase 3: back
    ax[0].vlines(seg['uturn end'] / freq, mi, ma, 'black', '--', linewidth=3)
    ax[0].add_patch(patches.Rectangle((seg['uturn end'] / freq, mi),  # (x,y)
                                      ((len(t) - 1) - seg['uturn end']) / freq,  # width
                                      ma - mi,  # height
                                      alpha=0.2, color="k"))
    ax[1].add_patch(patches.Rectangle((seg['uturn end'] / freq, 0),  # (x,y)
                                      ((len(t) - 1) - seg['uturn end']) / freq,  # width
                                      1,  # height
                                      alpha=0.2, color="k"))
    ax[1].text(seg['uturn end'] / freq + ((len(t) - 1) - seg['uturn end']) / (2 * freq), 0.5, 'straight (back)',
               fontsize=9,
               horizontalalignment='center', verticalalignment='center')

    ax[0].legend(loc="upper right")

    ax[1].set(ylabel='Phases')
    ax[1].set_yticks([])

    plt.show()


def find_intervals(pred, val):
  intervals = []
  in_interval = False

  for i, v in enumerate(pred):
    if v == val and not in_interval:
      start = i
      in_interval = True
    elif v != val and in_interval:
      end = i
      intervals.append((start, end))
      in_interval = False

  if in_interval:
    intervals.append((start, len(pred)))

  return intervals


def plot_gait_detection(y_true, y_pred, trial, path, title_base, sensor='LB', signal_channel='Acc_Z', process='raw', save_path=None):

    if process == 'raw':
        signal = pd.read_csv(f'{path}/{trial}_{process}_data_{sensor}.txt', sep='\t')
    elif process == 'processed':
        signal = pd.read_csv(f'{trial}_{process}_data.txt', sep='\t')

    data_plot = signal[[signal_channel]]

    red_patch   = mpatches.Patch(color="red",   alpha=0.3, label="Right gait")
    green_patch = mpatches.Patch(color="green", alpha=0.3, label="Left gait")
    blue_patch  = mpatches.Patch(color="blue",  alpha=0.1, label="No gait")

    fig, axes = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
    fig.suptitle(title_base, fontsize=12)

    for ax, y, subtitle in zip(axes, [y_true, y_pred], ["Ground Truth", "Predicted"]):
        for interval in find_intervals(y, 2):
            ax.axvspan(interval[0], interval[1], color="red", alpha=0.3)
        for interval in find_intervals(y, 1):
            ax.axvspan(interval[0], interval[1], color="green", alpha=0.3)
        for gap in find_intervals(y, 0):
            ax.axvspan(gap[0], gap[1], color="blue", alpha=0.1)

        ax.plot(data_plot.values, color='black', linewidth=0.8)
        ax.set_title(subtitle)
        ax.legend(handles=[red_patch, green_patch, blue_patch])
        ax.set_ylabel(signal_channel)

    axes[-1].set_xlabel("Sample")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Guardado en: {save_path}")

    plt.show()


def statistical_analysis(out_dir, model_name="", sensor=""):
    with open(os.path.join(out_dir, "summary.json")) as f:
        summary = json.load(f)

    df = pd.DataFrame(summary)
    df['group'] = df['subject'].str.extract(r'^([A-Za-z]+)')

    print(f"{'='*50}")
    print(f"Model: {model_name} | Sensor: {sensor}")
    print(f"{'='*50}")
    print(df[['val_f1', 'test_f1']].describe().round(4).to_string())
    print(f"\nTest F1 Median: {df['test_f1'].median():.4f}")
    print(f"Mean Test F1 : {df['test_f1'].mean():.4f} ± {df['test_f1'].std():.4f}")

    group_stats = (df.groupby('group')['test_f1']
                   .agg(['mean', 'median', 'std', 'count'])
                   .round(3)
                   .sort_values('mean'))
    print(f"\Per cohort:\n{group_stats.to_string()}")

    COLOR_MAP = {
        'HS':   ('Healthy',        'lightgreen'),
        'ACL':  ('Orthopaedic',    'lightcoral'),
        'HOA':  ('Orthopaedic',    'lightcoral'),
        'KOA':  ('Orthopaedic',    'lightcoral'),
        'CIPN': ('Neurological',   'lightblue'),
        'CVA':  ('Neurological',   'lightblue'),
        'PD':   ('Neurological',   'lightblue'),
        'RIL':  ('Neurological',   'lightblue'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Test F1 — {model_name}  |  {sensor}", fontsize=13, y=1.01)

    global_mean   = df['test_f1'].mean()
    global_median = df['test_f1'].median()
    threshold     = df['test_f1'].quantile(0.20)
    outliers      = df[df['test_f1'] < threshold]

    ax = axes[0]
    ax.boxplot(df['test_f1'], patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.5),
               medianprops=dict(color='red', linewidth=2), widths=0.4)
    ax.scatter([1]*len(df),       df['test_f1'],    alpha=0.3, color='steelblue', s=15, zorder=3)
    ax.scatter([1]*len(outliers), outliers['test_f1'], alpha=0.9, color='tomato', s=30, zorder=4, label='Worst 20%')
    ax.set_xticks([1]); ax.set_xticklabels(['All subjects'])
    ax.set_ylabel('Macro F1'); ax.set_title('Global distribution')
    ax.legend(); ax.grid(axis='y', alpha=0.4)

    ax = axes[1]
    ax.hist(df['test_f1'], bins=25, color='steelblue', alpha=0.75, edgecolor='white')
    ax.axvline(global_mean,   color='red',    linestyle='--', label=f'Mean {global_mean:.3f}')
    ax.axvline(global_median, color='orange', linestyle='--', label=f'Median {global_median:.3f}')
    ax.axvline(threshold,     color='tomato', linestyle=':',  label=f'P20 {threshold:.3f}')
    ax.set_xlabel('Macro F1'); ax.set_ylabel('Nº subjects')
    ax.set_title('Histogram'); ax.legend(); ax.grid(axis='y', alpha=0.4)

    ax = axes[2]
    seen_labels = set()
    for group, stats_row in group_stats.iterrows():
        label_name, color = COLOR_MAP.get(group, ('Other', 'gray'))
        label = label_name if label_name not in seen_labels else "_nolegend_"
        seen_labels.add(label_name)

        scores = df[df['group'] == group]['test_f1'].values
        bar = ax.bar(group, scores.mean(), yerr=scores.std(),
                     capsize=5, color=color, alpha=0.8, label=label)
        ax.bar_label(bar, fmt='%.3f', fontsize=8)

    ax.axhline(global_mean, color='black', linestyle='--', alpha=0.7, label=f'Global mean {global_mean:.3f}')
    ax.set_ylim(0.4, 1.1); ax.set_ylabel('Mean Macro F1')
    ax.set_title('By clinical cohort')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved in: {plot_path}")
