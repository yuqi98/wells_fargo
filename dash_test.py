# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import base64

#css dependency
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#read in csv
data = pd.read_csv('first_preprocess.csv', index_col='user_num')


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
    html.Button('User', id='button', style={'margin': '0px 0px 25px 5%'}),
    #User Details
    html.Div([html.Img(src=encode_image('user.png'), style={'height': '100px', 'boxShadow': '0px 0px 2px 2px rgba(204,204,204,0.4)',
                        'marginLeft': '15%', 'float': 'left'}),
            html.Div([html.P('Name: User1'), html.P('Location: Durham, North Carolina'), html.P('Area of most impact')], style={'padding': '2%', 'float': 'left'}),
            #increase current margin
            html.Div('A', style=styles['user_det'])
            ],
            style={'columnCount': 3}),
    html.Div(style={'float': 'clear'}),

    #plotted graphs
    html.Div(id="graphs", [
            dcc.Graph(id='personal-visual'),
            dcc.Graph(id='performance-chart'),
            dcc.Graph(id='group-visual')
            ],
            style={'columnCount': 3}),
    #recommendations
    html.H6('Recommendations', style={'color': 'SteelBlue', 'marginLeft': '2%'})

], style=styles['main'])

@app.callback(
    Output(component_id='graph', component_property='children'),
    [Input(component_id='input-box', component_property='value')]
)
def update_output_div(input_value):
    dat = data.copy()
    dat.columns = ['heating','bath','kitchen','TV','transportation','waste']
    df_pi = pd.DataFrame(index=data.index)
    df_pi['waste'] = dat['waste']
    df_pi['transportation'] = dat['transportation']
    df_pi['other'] = dat['heating'] + dat['bath'] + dat['kitchen'] + dat['TV']
    user = df_pi.iloc[input_value, :]
    labels = df_pi.columns
    values = user.values
    colors = ['#6B8E23', '#9ACD32','#98FB98']
    return figure={'data':[
                        go.Pie(labels=labels, values=values,
                               hoverinfo='label+percent', textinfo='value',
                               hole=.3,
                               textfont=dict(size=15),
                               marker=dict(colors=colors,
                               line=dict(color='#000000', width=0.5)))
                    ],
                    'layout': go.layout(
                        'title' = 'Person CFP Contribution',
                    )}

if __name__ == '__main__':
    app.run_server(debug=True)
