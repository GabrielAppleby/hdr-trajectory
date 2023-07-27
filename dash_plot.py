from pathlib import Path

import dash_bio as dashbio
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from config import CLUSTER_TEMPLATE, CLUSTER_OUTPUT_FILENAME_TEMPLATE, \
    PARSE_OUTPUT_FILENAME_TEMPLATE, UMAP_NN_PARAMS, UMAP_NN_PARAM_STEP, PROJECTION_TEMPLATE, MIN_CLUSTER_SIZE, \
    MIN_CLUSTER_SIZE_STEP, MIN_SAMPLES, MIN_SAMPLES_STEP, CLUSTER_SELECTION_EPSILON_STEP, CLUSTER_SELECTION_EPSILON

app = Dash(__name__)

def update_data(data_filename):
    MOLECULE_NAME: str = data_filename
    DATA_DIR: Path = Path('data')

    cluster_data = np.load(str(Path(DATA_DIR, MOLECULE_NAME, CLUSTER_OUTPUT_FILENAME_TEMPLATE.format(MOLECULE_NAME))))
    position_data = np.load(str(Path(DATA_DIR, MOLECULE_NAME, PARSE_OUTPUT_FILENAME_TEMPLATE.format(MOLECULE_NAME))))
    atom_names = position_data['atom_names']
    position_arr = position_data['arr']
    N_TIME_STEPS = position_arr.shape[1]
    return cluster_data, position_data, atom_names, position_arr, N_TIME_STEPS

cluster_data, position_data, atom_names, position_arr, N_TIME_STEPS = update_data('cyp')

def get_molecule_spec(traj_id, timestep):
    return [{
        'symbol': atom,
        'x': coords[0],
        'y': coords[1],
        'z': coords[2]
    } for atom, coords in zip(atom_names, position_arr[traj_id, timestep])]


app.layout = html.Div(id='app-entry', className='flex', children=[
    html.Div(id='left-panel', className='flex column center', children=[
        dcc.Dropdown(['cyp', 'dbh-parent', 'dbhmeo-7'], 'cyp', id='data-dropdown'),
        html.Div(id='num_neighbors_slider_div', children=[
            html.H4(children='Number of neighbors:'),
            dcc.Slider(id='num_neighbors_slider', className='dcc-slider-cludge', min=UMAP_NN_PARAMS[0], max=UMAP_NN_PARAMS[-1],
                       step=UMAP_NN_PARAM_STEP,
                       marks={nn: str(nn) for nn in UMAP_NN_PARAMS},
                       value=UMAP_NN_PARAMS[-1])],
                 className='slider flex column'),
        html.Div(id='min_cluster_size_div', children=[
            html.H4(children='Min cluster size:'),
            dcc.Slider(id='min_cluster_size_slider', className='dcc-slider-cludge', min=MIN_CLUSTER_SIZE[0], max=MIN_CLUSTER_SIZE[-1],
                       step=MIN_CLUSTER_SIZE_STEP,
                       marks={mse: str(mse) for mse in MIN_CLUSTER_SIZE},
                       value=MIN_CLUSTER_SIZE[-1])],
                 className='slider flex column'),
        html.Div(id='min_samples_div', children=[
            html.H4(children='Min samples:'),
            dcc.Slider(id='min_samples_slider', className='dcc-slider-cludge', min=MIN_SAMPLES[0], max=MIN_SAMPLES[-1],
                       step=MIN_SAMPLES_STEP,
                       marks={ms: str(ms) for ms in MIN_SAMPLES},
                       value=MIN_SAMPLES[-1])],
                 className='slider flex column'),
        html.Div(id='epsilon_slider_div', children=[
            html.H4(children='Selection epsilon:'),
            dcc.Slider(id='epsilon_slider', className='dcc-slider-cludge', min=CLUSTER_SELECTION_EPSILON[0], max=CLUSTER_SELECTION_EPSILON[-1],
                       step=CLUSTER_SELECTION_EPSILON_STEP,
                       value=CLUSTER_SELECTION_EPSILON[0])],
                 className='slider flex column'),
        html.Div(id='molecule_viewer', className='flex', children=[
            dashbio.Speck(
                id='speck',
                data=get_molecule_spec(0, 0),
                presetView='stickball',
                showLegend=True,
                view={'zoom': 0.05,
                      'translation': {'x': 6.0, 'y': 6.0},
                      'atomScale': 0.24,
                      'relativeAtomScale': 0.64,
                      'bondScale': 0.5,
                      'bonds': True,
                      'bondThreshold': 1.2,
                      },
            ),
        ]),
        html.Div(id='timestep_slider_div', children=[
            html.H4(children='Timestep:'),
            dcc.Slider(id='timestep_slider', className='dcc-slider-cludge', min=0, max=N_TIME_STEPS - 1,
                       step=50,
                       marks=None,
                       tooltip={"placement": "bottom", "always_visible": True},
                       value=N_TIME_STEPS - 1)],
                 className='slider flex column'),
    ]),
    html.Div(id='right-panel', className='flex column', children=[
        html.Div(id='projection', children=[
            dcc.Graph(
                id='projection_scatter',
                config={'modeBarButtonsToRemove': ['select', 'lasso']},
                style={'height': '100%'}
            ),
            dcc.Tooltip(id='projection_tooltip', style={'opacity': .8})
        ]),
    ])
])


def get_scatter(projection_data, cluster_data):
    fig = px.scatter(title='Product Overview', x=projection_data[:, 0], y=projection_data[:, 1], color=[i for i in cluster_data.astype('str')], custom_data=(list(range(projection_data.shape[0])),))
    fig.update_layout(
        {'title':
             {'font': {'size': 30},
              'x': 0.5,
              'xanchor': 'center'},
         'xaxis':
             {'title': {'text': 'umap x'}},
         'yaxis':
             {'title': {'text': 'umap y'}}
         }
    )
    return fig

@app.callback(
    Output(component_id='projection_scatter', component_property='figure'),
    Input(component_id='num_neighbors_slider', component_property='value'),
    Input(component_id='min_cluster_size_slider', component_property='value'),
    Input(component_id='min_samples_slider', component_property='value'),
    Input(component_id='epsilon_slider', component_property='value'),
    Input(component_id='data-dropdown', component_property='value'))
def update_scatter(n_neighbor, mcs, ms, e, data_filename):
    e = '{:.2f}'.format(e)
    cluster_data, position_data, atom_names, position_arr, N_TIME_STEPS = update_data(data_filename)
    return get_scatter(cluster_data[PROJECTION_TEMPLATE.format(n_neighbor)], cluster_data[CLUSTER_TEMPLATE.format(mcs, ms, e)])


@app.callback(
    Output('speck', 'data'),
    Output('projection_tooltip', 'show'),
    Output('projection_tooltip', 'bbox'),
    Output('projection_tooltip', 'children'),
    Input('projection_scatter', 'clickData'),
    Input(component_id='timestep_slider', component_property='value'),
    Input(component_id='data-dropdown', component_property='value')
    )
def update_molecule(click_data, timestep, data_filename):
    cluster_data, position_data, atom_names, position_arr, N_TIME_STEPS = update_data(data_filename)

    if click_data is not None:
        pt = click_data['points'][0]
        bbox = pt['bbox']
        traj_id = pt['customdata'][0]

        children = [
            html.Div(
                children=[
                    html.P('ID: {}'.format(traj_id)),
                ], style={'width': '70px', 'white-space': 'normal'})
        ]

        return get_molecule_spec(traj_id, timestep), True, bbox, children
    return get_molecule_spec(0, timestep), False, None, None


app.clientside_callback(
    """
    function() {
        const width = document.getElementById("left-panel").getBoundingClientRect().width - 80;
        return {'width': width, 'height': width}
    }
    """,
    Output('speck', 'style'),
    Input('molecule_viewer', 'style'),
)


if __name__ == '__main__':
    app.run_server(debug=True)
