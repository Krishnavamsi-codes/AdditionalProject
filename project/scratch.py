import pandas as pd
from visualization.plotter import Plotter

df = pd.read_csv('results/summary_results.csv')
plotter = Plotter('results')
runtime_df = df[['Method', 'Detection Time (ms)', 'Descriptor Time (ms)', 'Matching Time (ms)']].copy()
plotter.plot_runtimes(runtime_df, 'runtimes.png')
