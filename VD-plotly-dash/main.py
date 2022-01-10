import dash
from dash import dcc
from dash import html
import os
from dash.dependencies import Input, Output
import processing
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dashboard = processing.Dashboard()

# Create app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    # Title
    html.Div([
        html.H1(
            "Hearth Failure Prediction",
            style={"margin-bottom": "0px"},
        ),
        html.H2(
            "Visualización de Datos MACC - Práctica Final",
            style={"margin-bottom": "0px"},
        ),
        html.H4(
            "Asier Guerricagoitia, Andoni Martin, Markel Orallo, Eneko Vaquero",
            style={"margin-bottom": "0px"},
        ),
    ],
        id="title",
        className="one-half column",
    ),

    # dashboard
    html.Div([
        # filters
        html.Div([
            html.P("Select classification algorithm:"),
            dcc.Dropdown(
                id='algorithm-dropdown',
                clearable=False,
                options=[
                    {'label': 'SVC', 'value': 'SVC'},
                    {'label': 'NuSVC', 'value': 'NuSVC'},
                ],
                value='SVC'
            ),

            html.P("Select Cost:"),
            dcc.Slider(
                id='cost-slider',
                min=1,
                max=7,
                value=1,
                marks={i: "{}".format(10 ** i) for i in range(1, 8)},
            ),

            html.P("Select Nu:"),
            dcc.Slider(
                id='nu-slider',
                min=0.1,
                max=0.6,
                value=0.5,
                step=None,
                marks={
                    0.1: '0.1',
                    0.2: '0.2',
                    0.3: '0.3',
                    0.4: '0.4',
                    0.5: '0.5',
                    0.6: '0.6',
                },
            ),

            html.P("Select Kernel type:"),
            dcc.Dropdown(
                id='kernel-dropdown',
                clearable=False,
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'Poly', 'value': 'poly'},
                    {'label': 'rbf', 'value': 'rbf'},
                    {'label': 'Sigmoid', 'value': 'sigmoid'},
                ],
                value='rbf'
            ),

            html.P("Select number of clusters:"),
            dcc.Slider(
                id='clusters-slider',
                min=2,
                max=6,
                value=5,
                step=None,
                marks={
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                },
            ),
            html.P("Select clustering attributes:"),
            dcc.Dropdown(
                id='clustering-dropdown',
                options=dashboard.get_variable_names(),
                multi=True,
            )
        ],
            id="filters",
            className="container",
        ),
        # dashboards
        html.Div([
            html.Div([
                dcc.Graph(id='ROC-graph'),
            ],
                id="roc",
            ),
            html.Div([
                dcc.Graph(id='confusion-graph'),
            ],
                id="confusion",
            ),
        ],
            id="graphs",
        ),

        # dashboards
        html.Div([
            html.Div([
                dcc.Graph(id='last-graph'),
            ],
                id="last",
            ),

            html.Div([
                dcc.Graph(id='correlation-graph'),
            ],
                id="correlation",
            ),

        ],
            id="graphs2",
        ),
    ],

        id="dashboard",
    ),
])


@app.callback(Output('cost-slider', 'disabled'),
              [Input('algorithm-dropdown', 'value')])
def disable_cost_slider(algorithm):
    return algorithm != 'SVC'


@app.callback(Output('nu-slider', 'disabled'),
              [Input('algorithm-dropdown', 'value')])
def disable_nu_slider(algorithm):
    return algorithm != 'NuSVC'


@app.callback(
    Output("ROC-graph", "figure"),
    [Input("algorithm-dropdown", "value"),
     Input("cost-slider", "value"),
     Input("nu-slider", "value"),
     Input("kernel-dropdown", "value")],
)
def ROC_Updated(algorithms, c, nu, kernel):
    dashboard.update_model(algorithms, c, nu, kernel)
    roc = dashboard.serve_roc_curve()
    return roc


@app.callback(
    Output("confusion-graph", "figure"),
    [Input("algorithm-dropdown", "value"),
     Input("cost-slider", "value"),
     Input("nu-slider", "value"),
     Input("kernel-dropdown", "value")],
)
def Confusion_Updated(algorithms, c, nu, kernel):
    dashboard.update_model(algorithms, c, nu, kernel)
    confusion = dashboard.serve_pie_confusion_matrix()
    return confusion


@app.callback(
    Output("correlation-graph", "figure"),
    [Input("clusters-slider", "value"),
     Input("clustering-dropdown", "value")],
)
def correlation_updated(value, atribute):
    print(atribute)
    fig = {}
    if atribute is not None and len(atribute) == 3:
        dataset2 = processing.finaldf
        X = dataset2.iloc[:, 0:10]
        kmeans = KMeans(n_clusters=value, init="k-means++", max_iter=200, n_init=10, random_state=123)
        identified_clusters = kmeans.fit_predict(X)
        cols = dashboard.get_columns(atribute)
        data_with_clusters = dataset2.copy()
        data_with_clusters['Cluster'] = identified_clusters
        fig = px.scatter_3d(data_with_clusters, x=atribute[0], y=atribute[1], z=atribute[2],
                            color='Cluster', opacity=0.8, size=cols.iloc[:, 0], size_max=20,
                            title='Clusters with Kmeans')
    return fig


@app.callback(
    Output("last-graph", "figure"),
    [Input("algorithm-dropdown", "value"),
     Input("cost-slider", "value"),
     Input("nu-slider", "value"),
     Input("kernel-dropdown", "value")],
)
def bar_figure(algorithms, c, nu, kernel):
    dashboard.update_model(algorithms, c, nu, kernel)
    bar = dashboard.most_importance_features()
    return bar


if __name__ == '__main__':
    app.run_server(debug=True)
