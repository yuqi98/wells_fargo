# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.plotly as py
from plotly.graph_objs import *
from scipy.stats import rayleigh
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import sqlite3
import datetime as dt
import base64

#css dependency
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#read in csv
user_doc = pd.read_csv('first_preprocess.csv', index_col='user_num')
np.random.seed(42)
random_x = np.random.randint(1,101,100)
random_y = np.random.randint(1,101,100)


#image read in
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())

#create app
app = dash.Dash('User CFP', external_stylesheets=external_stylesheets)

#custom css
styles = {
    'title': {'textAlign': 'center', 'color': 'DeepSkyBlue'},
    'user_det': {'height': '90px', 'boxShadow': '0px 0px 2px 2px rgba(204,204,204,0.4',
                'float': 'left', 'textAlign': 'center', 'color': 'Red', 'fontSize': '60px', 'padding': '0px, 10px, 0px, 10px'},
    'main': {'borderRadius': '2px', 'padding': '5px 5px 5px 5px',
              'marginLeft': 'auto', 'marginRight': 'auto', "width": "95%",
              'boxShadow': '0px 0px 2px 2px rgba(204,204,204,0.4)'}
}

app.layout = html.Div([
    html.H1('User Carbon FootPrint Statistics', style=styles['title']),
    html.Div(dcc.Input(id='input-box', type='text', style={'margin': '0px 0px 0px 5%'})),
    html.Button('Submit', id='button', style={'margin': '0px 0px 25px 5%'}),
    #User Details
    html.Div([html.Img(src=encode_image('user.png'), style={'height': '100px', 'boxShadow': '0px 0px 2px 2px rgba(204,204,204,0.4)',
                        'marginLeft': '15%', 'float': 'left'}),
            html.Div([html.P('Name: User1'), html.P('Location: Durham, North Carolina'), html.P('Area of most impact')], style={'padding': '2%', 'float': 'left'}),
            html.Div('A', style=styles['user_det'])
            ],
            style={'columnCount': 3}),
    html.Div(style={'float': 'clear'}),

    #plotted graphs
    html.Div([
            dcc.Graph(id='group-visual',
                figure={
                    'data':[
                        {'x': user_doc.columns, 'y': user_doc.iloc[1], 'type': 'bar',
                        'align': 'center'}
                    ],
                    'layout': {'title': 'Group Performance'}
            }),
            dcc.Graph(
                id='scatter3',
                figure={
                    'data': [
                        go.Scatter(
                            x = random_x,
                            y = random_y,
                            mode = 'markers',
                            marker = {
                                'size': 12,
                                'color': 'rgb(51,204,153)',
                                'symbol': 'pentagon',
                                'line': {'width': 2}
                                }
                        )
                    ],
                    'layout': go.Layout(
                        title = 'ScatterPlot of AggregateCFP',
                        xaxis = {'title': 'Some random x-values'},
                        yaxis = {'title': 'Some random y-values'},
                        hovermode='closest'
                    )
                }
            ),
            dcc.Graph(id='third-visual',
                        figure={
                            'data':[
                                {'x': user_doc.columns, 'y': user_doc.iloc[50], 'type': 'bar',
                                'align': 'center'}
                            ],
                            'layout': {'title': 'Connections'}
            })
            ],
            style={'columnCount': 3}),
    #recommendations
    html.H6('Recommendations', style={'color': 'SteelBlue', 'marginLeft': '2%'})

], style=styles['main'])

# @app.callback(
#     Output(component_id='user-id', component_property='children'),
#     [Input(component_id='request-id', component_property='value')]
# )
# def update_output_div(input_value):
#     return '{}'.format(input_value)

if __name__ == '__main__':
    app.run_server(debug=True)
