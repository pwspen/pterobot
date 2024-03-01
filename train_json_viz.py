import pandas as pd
import plotly.graph_objects as go
import json

# Assuming the JSON structure is a dictionary with lists, convert it to a DataFrame
# If your JSON structure is different, you might need to adjust this part

# Function to create a plotly figure

def save_plotly_figure(fig, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fig.to_html())


def create_plotly_figure(file_path='train.json', save_path=None):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    df = pd.DataFrame.from_dict(json_data, orient='index')

    fig = go.Figure()
    
    default_visible_metrics = ['eval/episode_forward_reward',
                               'eval/episode_reward_alive',
                               'eval/episode_reward_quadctrl',
                               'eval/episode_vertical_reward',
                               'eval/episode_reward']

    # Add a trace for each column in the DataFrame
    for metric in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[metric], 
            mode='lines+markers', 
            name=metric,
            visible=True if metric in default_visible_metrics else 'legendonly'))
    
    # Update layout to add titles and enable legend to toggle traces
    fig.update_layout(
        title=file_path.replace('.json', ''),
        xaxis_title='Train Step',
        yaxis_title='Val',
        legend_title='Metrics',
        legend=dict(traceorder='normal', title_font_family='Arial, sans-serif'),
    )
    
    if not save_path:
        fig.show()
    else:
        save_plotly_figure(fig, save_path)

    return fig
