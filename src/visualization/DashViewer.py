import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go


class MeshVisualizer:
    def __init__(self, meshes):
        self.meshes = meshes
        self.app = dash.Dash(__name__)
        self.length = len(meshes)

    def run(self):
        # Define the layout of the Dash app
        self.app.layout = html.Div([
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
            mesh = self.meshes[mesh_to_show]

            x, y, z = mesh.tensor_points.T
            i, j, k = mesh.cells[0].data.T
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
                    aspectmode='cube'
                )
            )

            return mesh_figure

        self.app.run_server(host='localhost', port=8005)


class ProposalVisualizer:
    def __init__(self, mesh):
        self.mesh = mesh
        self.app = dash.Dash(__name__)
        self.length = mesh.batch_size

    def run(self):
        # Define the layout of the Dash app
        self.app.layout = html.Div([
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
                    aspectmode='cube'
                )
            )

            return mesh_figure

        self.app.run_server(host='localhost', port=8005)
