from train_eval import eval
from train_json_viz import create_plotly_figure, save_plotly_figure
from pathlib import Path

policy = 12
# create_plotly_figure(f'trained_policies/{policy}/train{policy}.json', f'trained_policies/{policy}/train{policy}.html')
eval(f'trained_policies/{policy}/policy{policy}.zip', 'pterobot')