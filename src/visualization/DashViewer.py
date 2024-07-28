import math

import dash
import meshio
import numpy as np
import torch
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
from flask import request

from src.mesh.TMesh import get_transformation_matrix, TorchMesh


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
    def __init__(self, app, model, target):
        self.app = app
        self.model = model
        self.target = target
        self.num_parameters = self.model.sample_size + 6
        self.parameters = np.zeros(self.num_parameters)
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [1.2 * np.min(self.model.mean), 1.2 * np.max(self.model.mean)]
        sliders = []
        translation_sliders = []
        rotation_sliders = []
        for i in range(self.num_parameters - 6):
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
        for i in range(3):
            translation_sliders.append(html.Label(f'Translation {i + 1}', style={'font-size': '10px'}))
            translation_sliders.append(
                dcc.Slider(
                    id=f'translation-{i}',
                    min=-100,
                    max=100,
                    step=0.2,
                    value=0,
                    marks={-100: '-100', 0: '0', 100: '100'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag',
                )
            )
            rotation_sliders.append(html.Label(f'Rotation {i + 1}', style={'font-size': '10px'}))
            rotation_sliders.append(
                dcc.Slider(
                    id=f'rotation-{i}',
                    min=-math.pi,
                    max=math.pi,
                    step=0.005,
                    value=0,
                    marks={-math.pi: '-3.14', 0: '0', math.pi: '3.14'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag',
                )
            )

        layout = html.Div([
            html.H1("3D Model Analyser", style={'text-align': 'center'}),
            html.Div([
                dcc.Graph(id='mesh-plot', style={'width': '100%', 'height': '80vh', 'display': 'inline-block'}),
                html.Div(sliders,
                         style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
                html.Div(translation_sliders,
                         style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
                html.Div(rotation_sliders,
                         style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
            ])
        ])

        inputs = [Input(f'param-{i}', 'value') for i in range(self.num_parameters - 6)]
        translation_inputs = [Input(f'translation-{i}', 'value') for i in range(3)]
        rotation_inputs = [Input(f'rotation-{i}', 'value') for i in range(3)]
        inputs.extend(translation_inputs)
        inputs.extend(rotation_inputs)

        @callback(
            Output('mesh-plot', 'figure'),
            inputs
        )
        def update_mesh(*parameters):
            params = np.asarray(parameters)
            translation = params[-6:-3]
            rotation = params[-3:]
            points = self.model.get_points_from_parameters(params[:-6])
            new_mesh = meshio.Mesh(points.astype(np.float32), [meshio.CellBlock('triangle',
                                                                                         self.target.cells[0].data.astype(
                                                                                             np.int64))])
            new_torch_mesh = TorchMesh(new_mesh, 'display')
            new_torch_mesh.apply_translation(translation)
            new_torch_mesh.apply_rotation(rotation)
            x, y, z = np.transpose(new_torch_mesh.points)
            i, j, k = new_torch_mesh.cells[0].data.T
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
        self.num_parameters = self.sampler.model.sample_size + 6
        self.parameters = self.sampler.proposal.chain
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [1.2 * np.min(self.sampler.model.mean), 1.2 * np.max(self.sampler.model.mean)]
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
                        step=1000,
                        value=0,
                        marks={i: str(i) for i in range(0, self.sampler.proposal.chain_length, 1000)}
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
            translation = params[-6:-3]
            rotation = params[-3:]
            points = self.sampler.model.get_points_from_parameters(params[:-6])
            new_mesh = meshio.Mesh(points.astype(np.float32), [meshio.CellBlock('triangle',
                                                                                self.sampler.target.cells[0].data.astype(
                                                                                    np.int64))])
            new_torch_mesh = TorchMesh(new_mesh, 'display')
            new_torch_mesh.apply_translation(translation)
            new_torch_mesh.apply_rotation(rotation)
            x, y, z = np.transpose(new_torch_mesh.points)
            i, j, k = new_torch_mesh.cells[0].data.T

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


def shutdown_app():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


class MainVisualizer:
    def __init__(self, mesh, model, sampler):
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.batch_mesh_visualizer = BatchMeshVisualizer(self.app, mesh)
        self.model_visualizer = ModelVisualizer(self.app, model, sampler.target)
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
        self.app.run_server(debug=False)

    def shutdown(self):
        @self.app.server.route('/shutdown', methods=['POST'])
        def shut():
            shutdown_app()
            return 'Dash-Server shutting down...'
