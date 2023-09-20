import matplotlib.pyplot as plt
import os
import pandas as pd


def get_min_max_interval(df):
    """
    Return df with min, median and max time intervals 
    (milliseconds) between accelerometer readings.
    """
    pids = []
    mins = []
    meds = []
    maxs = []
    for pid in list(df.pid.unique()):
        temp = df.loc[(df.pid==pid) & (df.time!=0)].reset_index(drop=True)
        temp = temp.sort_values('time', ascending=True)
        interval = temp[['time']].diff(axis=0)
        interval = interval.loc[interval.time.notna()] 
        pids.append(pid)
        mins.append(interval.time.min())
        meds.append(interval.time.median())
        maxs.append(interval.time.max())
    return pd.DataFrame(zip(pids,mins,meds,maxs), 
                        columns=['pid', 'Min Time Interval (ms)', 
                                 'Median Time Interval (ms)', 
                                 'Max Time Interval (ms)'])


def plot_acc_readings(df, pid):
    """
    Plot x, y, z accelerations for given pid over time.
    Save plot.
    """
    # Create folder for plots if does not exist.
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # Limit to data for give pid.
    temp = df.loc[(df.pid==pid) & (df.time!=0)].reset_index(drop=True)
    temp['datetime'] = pd.to_datetime(temp['time'], unit='ms')

    plt.figure(figsize=(20,10))
    plt.subplot(3, 1, 1)
    plt.title(f"Accelerometer Data for {pid}")
    plt.plot(temp.datetime, temp.x, '.-')
    plt.ylabel('X acceleration')

    plt.subplot(3, 1, 2)
    plt.plot(temp.datetime, temp.y, '.-')
    plt.ylabel('Y acceleration')

    plt.subplot(3, 1, 3)
    plt.plot(temp.datetime, temp.z, '.-')
    plt.xlabel('Datetime (ms)')
    plt.ylabel('Z acceleration')

    plt.savefig('plots/acc_readings.png')
    plt.show()


def plot_tac_readings(tac):
    """
    Plot raw TAC_Readings by subject pid over datetime.
    Save plot.
    """
    # Create folder for plots if does not exist.
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # Set plot size.
    plt.figure(figsize=(20,5))
    # Create datetime for reference in plot.
    tac['datetime'] = pd.to_datetime(tac['timestamp'], unit='s')
    for pid, group in tac.groupby("pid"):
        plt.plot(group["datetime"], group["TAC_Reading"], marker="o", 
            linestyle="", label=pid)
    plt.axhline(y=0.08, color='r', linestyle='-')
    plt.title("TAC Readings Over Time By Subject")
    plt.xlabel("Datetime")
    plt.legend()
    plt.savefig('plots/tac_readings.png')
    plt.show()


def plot_class_balance(tac, title):
    """
    Plot class frequency and percentage.
    Save plot.
    """
     # Create folder for plots if does not exist.
    if not os.path.exists('plots'):
        os.makedirs('plots')
    intox = tac[['intoxicated','pid']].groupby('intoxicated').\
        agg({'pid':'count'}).reset_index()
    intox['intoxicated'] = intox['intoxicated'].astype('int').astype('str')
    plt.bar(intox.intoxicated, height=intox.pid)
    # Percent labels.
    total = intox.pid.sum()
    for i, v in enumerate(intox.pid):
        plt.text(i - .08, v -(total/10), str("{:.0%}".format(v/total)), 
                 fontsize = 12, fontweight='bold')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.savefig(f"plots/{title.lower().replace(' ','_')}.png")
    plt.show()