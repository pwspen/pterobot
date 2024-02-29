from train_eval import eval
from train_json_viz import create_plotly_figure

create_plotly_figure('trained_policies/main_seq/train4.json')
# eval('trained_policies/main_seq/policy4.zip', 'pterobot')