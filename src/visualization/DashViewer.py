import math

import dash
import meshio
import torch
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go

from src.mesh import TorchMeshGpu


class BatchMeshVisualizer:
    def __init__(self, app, mesh, observed):
        self.app = app
        self.mesh = mesh
        self.length = mesh.batch_size
        self.observed = observed
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [torch.min(self.mesh.tensor_points), torch.max(self.mesh.tensor_points)]

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
            observed_com = torch.sum(self.mesh.tensor_points[self.observed], dim=0) / torch.sum(self.observed).item()
            x, y, z = self.mesh.tensor_points[:, :, mesh_to_show].T
            i, j, k = self.mesh.cells[0].data.T
            intensity = z
            # if mesh_to_show == 0:
            #     x_1, y_1, z_1 = self.mesh.tensor_points[:, :, 0].T
            #     x_2, y_2, z_2 = self.mesh.tensor_points[:, :, 1].T
            #     x_3, y_3, z_3 = self.mesh.tensor_points[:, :, 2].T
            #    i, j, k = self.mesh.cells[0].data.T
                # intensity = z
                # intensity_1 = z_1
                # intensity_2 = z_2
                # intensity_3 = z_3

            bone = [
                [0, 'rgb(240, 240, 240)'],  # Light grey
                [0.5, 'rgb(200, 200, 200)'],  # Medium grey
                [1, 'rgb(160, 160, 160)']  # Dark grey
            ]

                # observed_tris = self.observed[i] & self.observed[j] & self.observed[k]
                # other_tris = ~observed_tris
            mesh_figure = go.Figure(data=[
                go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                intensity=intensity,
                colorscale=bone,
                opacity=1.0,
                )
            ])

            # mesh_figure = go.Figure()

                # mesh_figure.update_layout(
                #     scene=dict(
                #         aspectmode='cube',
                #         xaxis=dict(nticks=10, range=display_range),
                #         yaxis=dict(nticks=10, range=display_range),
                #         zaxis=dict(nticks=10, range=display_range),
                #     )
                # )

                # mesh_figure.add_trace(
                #     go.Mesh3d(
                #         x=x_1, y=y_1, z=z_1,
                #         i=i, j=j, k=k,
                #         intensity=intensity_1,
                #         colorscale=bone,
                #         opacity=0.5,
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Mesh3d(
                #         x=x_2, y=y_2, z=z_2,
                #         i=i, j=j, k=k,
                #         intensity=intensity_2,
                #         colorscale=bone,
                #         opacity=0.5,
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Mesh3d(
                #         x=x_3, y=y_3, z=z_3,
                #         i=i, j=j, k=k,
                #         intensity=intensity_3,
                #         colorscale=bone,
                #         opacity=0.5,
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Scatter3d(
                #         x=[self.mesh.initial_com[:, 0][0].item()],
                #         y=[self.mesh.initial_com[:, 0][1].item()],
                #         z=[self.mesh.initial_com[:, 0][2].item()],
                #         mode='markers',
                #         marker=dict(size=12, color='red', symbol='cross'),
                #         name="Center of Mass"
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Scatter3d(
                #         x=[self.mesh.initial_com[:, 1][0].item()],
                #         y=[self.mesh.initial_com[:, 1][1].item()],
                #         z=[self.mesh.initial_com[:, 1][2].item()],
                #         mode='markers',
                #         marker=dict(size=12, color='red', symbol='cross'),
                #         name="Center of Mass"
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Scatter3d(
                #         x=[self.mesh.initial_com[:, 2][0].item()],
                #         y=[self.mesh.initial_com[:, 2][1].item()],
                #         z=[self.mesh.initial_com[:, 2][2].item()],
                #         mode='markers',
                #         marker=dict(size=12, color='red', symbol='cross'),
                #         name="Center of Mass"
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Scatter3d(
                #         x=[observed_com[:, 0][0].item()],
                #         y=[observed_com[:, 0][1].item()],
                #         z=[observed_com[:, 0][2].item()],
                #         mode='markers',
                #         marker=dict(size=12, color='black', symbol='cross'),
                #         name="Center of Mass"
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Scatter3d(
                #         x=[observed_com[:, 1][0].item()],
                #         y=[observed_com[:, 1][1].item()],
                #         z=[observed_com[:, 1][2].item()],
                #         mode='markers',
                #         marker=dict(size=12, color='black', symbol='cross'),
                #         name="Center of Mass"
                #     )
                # )
                #
                # mesh_figure.add_trace(
                #     go.Scatter3d(
                #         x=[observed_com[:, 2][0].item()],
                #         y=[observed_com[:, 2][1].item()],
                #         z=[observed_com[:, 2][2].item()],
                #         mode='markers',
                #         marker=dict(size=12, color='black', symbol='cross'),
                #         name="Center of Mass"
                #     )
                # )

                #
                # mesh_figure.add_trace(
                #    go.Mesh3d(
                #         x=x, y=y, z=z,
                #         i=i[~observed_tris], j=j[~observed_tris], k=k[~observed_tris],
                #         intensity=intensity,
                #         colorscale=bone,
                #         opacity=0.3,
                #     )
                # )

            mesh_figure.update_layout(
                scene=dict(
                aspectmode='data',  # Adjust aspect ratio to the data
                xaxis=dict(visible=False),  # Hide x-axis
                    yaxis=dict(visible=False),  # Hide y-axis
                    zaxis=dict(visible=False),  # Hide z-axis
                    xaxis_showgrid=False,  # Disable x-axis grid
                    yaxis_showgrid=False,  # Disable y-axis grid
                    zaxis_showgrid=False,  # Disable z-axis grid
                    bgcolor='rgba(0,0,0,0)'  # Transparent background
                ),
                showlegend=False  # Hide legend
            )

            return mesh_figure

            # else:
            #     # x, y, z = self.mesh.tensor_points[:, :, mesh_to_show].T
            #     x_1, y_1, z_1 = self.mesh.tensor_points[:, :, 3].T
            #     x_2, y_2, z_2 = self.mesh.tensor_points[:, :, 4].T
            #     x_3, y_3, z_3 = self.mesh.tensor_points[:, :, 5].T
            #     i, j, k = self.mesh.cells[0].data.T
            #     # intensity = z
            #     intensity_1 = z_1
            #     intensity_2 = z_2
            #     intensity_3 = z_3
            #     bone = [
            #         [0, 'rgb(240, 240, 240)'],  # Light grey
            #         [0.5, 'rgb(200, 200, 200)'],  # Medium grey
            #         [1, 'rgb(160, 160, 160)']  # Dark grey
            #     ]
            #
            #     observed_tris = self.observed[i] & self.observed[j] & self.observed[k]
            #     other_tris = ~observed_tris
            #     # mesh_figure = go.Figure(data=[
            #     #     go.Mesh3d(
            #     #         x=x,
            #     #         y=y,
            #     #         z=z,
            #     #         i=i,
            #     #         j=j,
            #     #         k=k,
            #     #         intensity=intensity,
            #     #         colorscale=bone,
            #     #         opacity=1.0,
            #     #     )
            #     # ])
            #
            #     mesh_figure = go.Figure()
            #
            #     # mesh_figure.update_layout(
            #     #     scene=dict(
            #     #         aspectmode='cube',
            #     #         xaxis=dict(nticks=10, range=display_range),
            #     #         yaxis=dict(nticks=10, range=display_range),
            #     #         zaxis=dict(nticks=10, range=display_range),
            #     #     )
            #     # )
            #
            #     mesh_figure.add_trace(
            #         go.Mesh3d(
            #             x=x_1, y=y_1, z=z_1,
            #             i=i, j=j, k=k,
            #             intensity=intensity_1,
            #             colorscale=bone,
            #             opacity=0.5,
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Mesh3d(
            #             x=x_2, y=y_2, z=z_2,
            #             i=i, j=j, k=k,
            #             intensity=intensity_2,
            #             colorscale=bone,
            #             opacity=0.5,
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Mesh3d(
            #             x=x_3, y=y_3, z=z_3,
            #             i=i, j=j, k=k,
            #             intensity=intensity_3,
            #             colorscale=bone,
            #             opacity=0.5,
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Scatter3d(
            #             x=[self.mesh.initial_com[:, 3][0].item()],
            #             y=[self.mesh.initial_com[:, 3][1].item()],
            #             z=[self.mesh.initial_com[:, 3][2].item()],
            #             mode='markers',
            #             marker=dict(size=12, color='red', symbol='cross'),
            #             name="Center of Mass"
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Scatter3d(
            #             x=[self.mesh.initial_com[:, 4][0].item()],
            #             y=[self.mesh.initial_com[:, 4][1].item()],
            #             z=[self.mesh.initial_com[:, 4][2].item()],
            #             mode='markers',
            #             marker=dict(size=12, color='red', symbol='cross'),
            #             name="Center of Mass"
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Scatter3d(
            #             x=[self.mesh.initial_com[:, 5][0].item()],
            #             y=[self.mesh.initial_com[:, 5][1].item()],
            #             z=[self.mesh.initial_com[:, 5][2].item()],
            #             mode='markers',
            #             marker=dict(size=12, color='red', symbol='cross'),
            #             name="Center of Mass"
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Scatter3d(
            #             x=[observed_com[:, 3][0].item()],
            #             y=[observed_com[:, 3][1].item()],
            #             z=[observed_com[:, 3][2].item()],
            #             mode='markers',
            #             marker=dict(size=12, color='black', symbol='cross'),
            #             name="Center of Mass"
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Scatter3d(
            #             x=[observed_com[:, 4][0].item()],
            #             y=[observed_com[:, 4][1].item()],
            #             z=[observed_com[:, 4][2].item()],
            #             mode='markers',
            #             marker=dict(size=12, color='black', symbol='cross'),
            #             name="Center of Mass"
            #         )
            #     )
            #
            #     mesh_figure.add_trace(
            #         go.Scatter3d(
            #             x=[observed_com[:, 5][0].item()],
            #             y=[observed_com[:, 5][1].item()],
            #             z=[observed_com[:, 5][2].item()],
            #             mode='markers',
            #             marker=dict(size=12, color='black', symbol='cross'),
            #             name="Center of Mass"
            #         )
            #     )
            #
            #     #
            #     # mesh_figure.add_trace(
            #     #    go.Mesh3d(
            #     #         x=x, y=y, z=z,
            #     #         i=i[~observed_tris], j=j[~observed_tris], k=k[~observed_tris],
            #     #         intensity=intensity,
            #     #         colorscale=bone,
            #     #         opacity=0.3,
            #     #     )
            #     # )
            #
            #     mesh_figure.update_layout(
            #         scene=dict(
            #             aspectmode='data',  # Adjust aspect ratio to the data
            #             xaxis=dict(visible=False),  # Hide x-axis
            #             yaxis=dict(visible=False),  # Hide y-axis
            #             zaxis=dict(visible=False),  # Hide z-axis
            #             xaxis_showgrid=False,  # Disable x-axis grid
            #             yaxis_showgrid=False,  # Disable y-axis grid
            #             zaxis_showgrid=False,  # Disable z-axis grid
            #             bgcolor='rgba(0,0,0,0)'  # Transparent background
            #         ),
            #         showlegend=False  # Hide legend
            #     )
            #
            #     return mesh_figure

        return layout


class ModelVisualizer:
    def __init__(self, app, model, batched_ref):
        self.app = app
        self.model = model
        self.batched_ref = batched_ref
        # Make sure all components are on the cpu, i.e., model, proposal, meshes.
        self.num_parameters = self.model.rank + 6
        self.parameters = torch.zeros(self.num_parameters)
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [1.2 * torch.min(self.model.mean), 1.2 * torch.max(self.model.mean)]
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
                html.Button('Draw Random Sample', id='randomize-button', n_clicks=0,
                            style={'display': 'block', 'margin': '20px auto'}),
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

        mean_points = self.model.get_points_from_parameters(torch.zeros((self.num_parameters - 6), device=self.model.dev))
        mean_mesh = meshio.Mesh(mean_points.cpu().numpy(), [meshio.CellBlock('triangle',
                                                                           self.batched_ref.cells[
                                                                               0].data.cpu().numpy())])
        mean_torch_mesh = TorchMeshGpu(mean_mesh, 'display', torch.device("cpu"))
        x_mean, y_mean, z_mean = mean_torch_mesh.tensor_points.T

        @callback(
            Output('mesh-plot', 'figure'),
            inputs
        )
        def update_mesh(*parameters):
            params = torch.tensor(parameters, dtype=torch.float32)
            translation = params[-6:-3]
            rotation = params[-3:]
            points = self.model.get_points_from_parameters(params[:-6])
            new_mesh = meshio.Mesh(points.cpu().numpy(), [meshio.CellBlock('triangle',
                                                                           self.batched_ref.cells[
                                                                               0].data.cpu().numpy())])
            new_torch_mesh = TorchMeshGpu(new_mesh, 'display', torch.device("cpu"))
            new_torch_mesh.apply_translation(translation)
            new_torch_mesh.apply_rotation(rotation)
            x, y, z = new_torch_mesh.tensor_points.T
            i_, j, k = new_torch_mesh.cells[0].data.T
            intensity = z
            # mesh_figure = go.Figure(data=[
            #     go.Mesh3d(
            #         x=x,
            #         y=y,
            #         z=z,
            #         i=i_,
            #         j=j,
            #         k=k,
            #         color='lightpink',
            #         opacity=0.50
            #     )
            # ])

            bone = [
                [0, 'rgb(240, 240, 240)'],  # Light grey
                [0.5, 'rgb(200, 200, 200)'],  # Medium grey
                [1, 'rgb(160, 160, 160)']  # Dark grey
            ]

            mesh_figure = go.Figure()
            mesh_figure.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i_, j=j, k=k,
                    # intensity=intensity,
                    color='lightpink',
                    opacity=0.8,
                )
            )

            mesh_figure.add_trace(
                go.Mesh3d(
                    x=x_mean, y=y_mean, z=z_mean,
                    i=i_, j=j, k=k,
                    intensity=intensity,
                    colorscale=bone,
                    opacity=0.3,
                )
            )



            mesh_figure.update_layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(nticks=10, range=display_range),
                    yaxis=dict(nticks=10, range=display_range),
                    zaxis=dict(nticks=10, range=display_range),
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    zaxis_showgrid=False,
                    xaxis_showbackground=False,
                    yaxis_showbackground=False,
                    zaxis_showbackground=False,
                    bgcolor='rgba(0,0,0,0)'
                )
            )

            # mesh_figure.update_layout(
            #     scene=dict(
            #         aspectmode='data',  # Adjust aspect ratio to the data
            #         xaxis=dict(nticks=10, range=display_range),  # Hide x-axis
            #         yaxis=dict(nticks=10, range=display_range),  # Hide y-axis
            #         zaxis=dict(nticks=10, range=display_range),  # Hide z-axis
            #         xaxis_showgrid=False,  # Disable x-axis grid
            #         yaxis_showgrid=False,  # Disable y-axis grid
            #         zaxis_showgrid=False,  # Disable z-axis grid
            #         bgcolor='rgba(0,0,0,0)'  # Transparent background
            #     ),
            #     showlegend=False  # Hide legend
            # )

            return mesh_figure

        @callback(
            [Output(f'param-{i}', 'value') for i in range(self.num_parameters - 6)],
            Input('randomize-button', 'n_clicks')
        )
        def randomize_parameters(n_clicks):
            if n_clicks > 0:
                random_params = torch.randn(self.num_parameters - 6).tolist()
                return random_params
            else:
                return [0] * (self.num_parameters - 6)

        return layout


class ChainVisualizer:
    def __init__(self, app, sampler):
        self.app = app
        self.sampler = sampler
        # Make sure all components are on the cpu, i.e., model, proposal, meshes.
        self.num_parameters = self.sampler.model.rank + 6
        self.parameters = self.sampler.proposal.chain
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        display_range = [1.2 * torch.min(self.sampler.model.mean), 1.2 * torch.max(self.sampler.model.mean)]
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
                        max=self.sampler.proposal.chain_length,
                        step=1000,
                        value=1,
                        marks={**{i: str(i) for i in range(1000, self.sampler.proposal.chain_length + 1, 1000)},
                               **{1: '1'}}
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
            params = self.parameters[:, batch, chain_element - 1]
            if chain_element > 0:
                translation = params[-6:-3]
                rotation = params[-3:]
                points = self.sampler.model.get_points_from_parameters(params[:-6])
                new_mesh = meshio.Mesh(points.numpy(), [meshio.CellBlock('triangle', self.sampler.batch_mesh.cells[
                    0].data.numpy())])
                new_torch_mesh = TorchMeshGpu(new_mesh, 'display', torch.device("cpu"))
                new_torch_mesh.apply_rotation(rotation)
                new_torch_mesh.apply_translation(translation)
            else:
                points = self.sampler.model.get_points_from_parameters(torch.zeros_like(params[:-6]))
                new_mesh = meshio.Mesh(points.numpy(), [meshio.CellBlock('triangle', self.sampler.batch_mesh.cells[
                    0].data.numpy())])
                new_torch_mesh = TorchMeshGpu(new_mesh, 'display', torch.device("cpu"))

            x, y, z = new_torch_mesh.tensor_points.T
            i, j, k = new_torch_mesh.cells[0].data.T

            x_ref, y_ref, z_ref = self.sampler.target_points[:, :, batch].T
            i_ref, j_ref, k_ref = self.sampler.target.cells[0].data.T

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
                i=i_ref,
                j=j_ref,
                k=k_ref,
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


class PosteriorVisualizer:
    def __init__(self, app, proposal):
        self.app = app
        self.proposal = proposal
        self.dev = self.proposal.dev
        self.posterior = proposal.posterior
        self.batch_size = proposal.batch_size
        self.layout = self.setup_layout_and_callbacks()

    def setup_layout_and_callbacks(self):
        layout = html.Div([
            html.H1("Trace Plots (Log Density Values)", style={'textAlign': 'center', 'marginTop': '10px'}),
            dcc.Graph(id='density-plot'),
            html.Div([
                html.Label('Remove a manually defined burn-in period from the density plot: '),
                dcc.Input(
                    id='x-axis-start',
                    type='number',
                    value=0,
                    min=0,
                    max=self.proposal.chain_length - 1,
                    step=1
                )
            ], style={'margin': '10px'}),
            dcc.Slider(
                id='slider',
                min=0,
                max=self.batch_size - 1,
                step=1,
                value=0,
                marks={i: str(i) for i in range(self.batch_size)}
            )
        ])

        @callback(
            Output('density-plot', 'figure'),
            Input('slider', 'value'),
            Input('x-axis-start', 'value')
        )
        def update_figure(value, axis_start):
            if axis_start is None or axis_start < 0:
                axis_start = 0
            x, y = torch.arange(axis_start, self.proposal.chain_length, 1), self.posterior[value,
                                                                            axis_start:self.proposal.chain_length]
            figure = {
                'data': [go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(size=5))],
                'layout': go.Layout(
                    title=f'Trace Plot for Batch {value}',
                    xaxis={'title': 'Iteration'},
                    yaxis={'title': 'Log Density Values'}
                )
            }
            return figure

        return layout


class MainVisualizer:
    def __init__(self, mesh, model, sampler, observed):
        self.app = dash.Dash(__name__, suppress_callback_exceptions=False)
        self.batch_mesh_visualizer = BatchMeshVisualizer(self.app, mesh, observed)
        self.model_visualizer = ModelVisualizer(self.app, model, mesh)
        self.chain_visualizer = ChainVisualizer(self.app, sampler)
        self.posterior_visualizer = PosteriorVisualizer(self.app, sampler.proposal)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            dcc.Dropdown(
                id='scene-dropdown',
                options=[
                    {'label': 'Batch Mesh', 'value': 'scene1'},
                    {'label': 'Model', 'value': 'scene2'},
                    {'label': 'Markov Chain', 'value': 'scene3'},
                    {'label': 'Log-Density Posterior', 'value': 'scene4'}
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
            elif scene == 'scene4':
                return self.posterior_visualizer.layout
            return html.Div()

    def run(self):
        self.app.run_server(debug=False)


class VisualizerLight:
    def __init__(self, mesh):
        self.app = dash.Dash(__name__, suppress_callback_exceptions=False)
        self.mesh_visualizer = BatchMeshVisualizer(self.app, mesh)
        self.setup_layout()

    def setup_layout(self):
        self.app.layout = self.mesh_visualizer.layout

    def run(self):
        self.app.run_server(debug=False)
