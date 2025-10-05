# src/06_plot_utils.py
import matplotlib.pyplot as plt

def plot_time_series(series, title, ylabel="Value", xlabel="Time", savepath=None):
    plt.figure(figsize=(12,5))
    series.plot()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)
    if savepath:
        plt.savefig(savepath)
    plt.show()
