import dash
import numpy as np
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go


class BatchMeshVisualizer:
    def __init__(self, app, mesh):
        self.app = app
        self.mesh = mesh
        self.length = mesh.batch_size
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [np.min(self.mesh.points), np.max(self.mesh.points)]

        # Define the layout of the Dash app
        layout = html.Div([
            html.Div([
                html.H1("3D Mesh Visualization", style={'textAlign': 'center', 'marginTop': '10px'}),
                dcc.Slider(
                    0,
                    self.length - 1,
                    step=1,
                    value=0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='custom-slider'
                )
            ], style={'height': '15vh', 'width': '100%'}),

            html.Div([
                dcc.Graph(
                    id='3d-mesh',
                    style={'width': '100%', 'height': '85vh'}
                )
            ], style={'height': '85vh', 'width': '100%'})
        ], style={'height': '100vh', 'display': 'flex', 'flex-direction': 'column'})

        @callback(
            Output('3d-mesh', 'figure'),
            Input('custom-slider', 'value'))
        def update_figure(mesh_to_show):
            x, y, z = self.mesh.tensor_points[:, :, mesh_to_show].T
            i, j, k = self.mesh.cells[0].data.T
            mesh_figure = go.Figure(data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    color='lightpink',
                    opacity=0.50
                )
            ])

            mesh_figure.update_layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(nticks=10, range=display_range),
                    yaxis=dict(nticks=10, range=display_range),
                    zaxis=dict(nticks=10, range=display_range),
                )
            )

            return mesh_figure

        return layout


class ModelVisualizer:
    def __init__(self, app, model):
        self.app = app
        self.model = model
        self.num_parameters = self.model.sample_size
        self.parameters = np.zeros(self.num_parameters)
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [np.min(self.model.meshes[0].points), np.max(self.model.meshes[0].points)]
        sliders = []
        for i in range(self.num_parameters):
            sliders.append(html.Label(f'Parameter {i + 1}', style={'font-size': '10px'}))
            sliders.append(
                dcc.Slider(
                    id=f'param-{i}',
                    min=-5,
                    max=5,
                    step=0.01,
                    value=0,
                    marks={-5: '-5', 0: '0', 5: '5'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag',
                )
            )

        layout = html.Div([
            html.H1("3D Model Analyser", style={'text-align': 'center'}),
            html.Div([
                dcc.Graph(id='mesh-plot', style={'width': '100%', 'height': '80vh', 'display': 'inline-block'}),
                html.Div(sliders,
                         style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
            ])
        ])

        inputs = [Input(f'param-{i}', 'value') for i in range(self.num_parameters)]

        @callback(
            Output('mesh-plot', 'figure'),
            inputs
        )
        def update_mesh(*parameters):
            params = np.asarray(parameters)
            x, y, z = np.transpose(self.model.get_points_from_parameters(params))
            i, j, k = self.model.meshes[0].cells[0].data.T
            mesh_figure = go.Figure(data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    color='lightpink',
                    opacity=0.50
                )
            ])

            mesh_figure.update_layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(nticks=10, range=display_range),
                    yaxis=dict(nticks=10, range=display_range),
                    zaxis=dict(nticks=10, range=display_range),
                )
            )

            return mesh_figure

        return layout


class ChainVisualizer:
    def __init__(self, app, sampler):
        self.app = app
        self.sampler = sampler
        self.num_parameters = self.sampler.model.sample_size
        self.parameters = self.sampler.proposal.chain
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [np.min(self.sampler.model.meshes[0].points), np.max(self.sampler.model.meshes[0].points)]
        layout = html.Div([
            html.H1("3D Markov Chain Viewer", style={'text-align': 'center'}),
            html.Div([
                dcc.Graph(id='mesh-plot', style={'width': '80%', 'height': '70vh'}),
                html.Div([
                    html.Label('Select Batch'),
                    dcc.Slider(
                        id='batch-slider',
                        min=0,
                        max=self.sampler.batch_size - 1,
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(self.sampler.batch_size)}
                    ),
                    html.Label('Select Chain Element'),
                    dcc.Slider(
                        id='chain-slider',
                        min=0,
                        max=self.sampler.proposal.chain_length - 1,
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(self.sampler.proposal.chain_length)}
                    )
                ], style={'width': '100%', 'padding': '10px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})

        ])

        # Callback to update the 3D mesh plot
        @callback(
            Output('mesh-plot', 'figure', allow_duplicate=True),
            [Input('batch-slider', 'value'),

             Input('chain-slider', 'value')],
            prevent_initial_call=True
        )
        def update_display(batch, chain_element):
            params = np.asarray(self.parameters[:, batch, chain_element])
            x, y, z = np.transpose(self.sampler.model.get_points_from_parameters(params))
            i, j, k = self.sampler.model.meshes[0].cells[0].data.T

            x_ref, y_ref, z_ref = np.transpose(self.sampler.target_points)

            mesh_figure = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color='limegreen',
                opacity=0.70
            )

            ref_figure = go.Mesh3d(
                x=x_ref,
                y=y_ref,
                z=z_ref,
                i=i,
                j=j,
                k=k,
                color='darkmagenta',
                opacity=0.30
            )

            mesh_layout = go.Layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(nticks=10, range=display_range),
                    yaxis=dict(nticks=10, range=display_range),
                    zaxis=dict(nticks=10, range=display_range),
                ),
                showlegend=True
            )

            return go.Figure(data=[mesh_figure, ref_figure], layout=mesh_layout)

        return layout


class MainVisualizer:
    def __init__(self, mesh, model, sampler):
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.batch_mesh_visualizer = BatchMeshVisualizer(self.app, mesh)
        self.model_visualizer = ModelVisualizer(self.app, model)
        self.chain_visualizer = ChainVisualizer(self.app, sampler)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            dcc.Dropdown(
                id='scene-dropdown',
                options=[
                    {'label': 'Batch Mesh', 'value': 'scene1'},
                    {'label': 'Model', 'value': 'scene2'},
                    {'label': 'Markov Chain', 'value': 'scene3'}
                ],
                value='scene1'
            ),
            html.Div(id='scene-container')
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('scene-container', 'children'),
            [Input('scene-dropdown', 'value')]
        )
        def display_scene(scene):
            if scene == 'scene1':
                return self.batch_mesh_visualizer.layout
            elif scene == 'scene2':
                return self.model_visualizer.layout
            elif scene == 'scene3':
                return self.chain_visualizer.layout
            return html.Div()

    def run(self):
        self.app.run_server(debug=True)
