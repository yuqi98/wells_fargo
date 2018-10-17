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
from sklearn.preprocessing import MinMaxScaler
import base64

#css dependency
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#read user_data
data = pd.read_csv('first_preprocess.csv', index_col='user_num')

#read products_data
products = pd.read_csv('products.csv', index_col='group')
products = products.iloc[:, ~products.columns.str.contains('^Unnamed')]

#read tips_data
#tips = pd.read_csv('tips.csv')

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
              'boxShadow': '0px 0px 2px 2px rgba(204,204,204,0.4)',
              },
}

app.layout = html.Div([
    html.H1('User Carbon FootPrint Statistics', style=styles['title']),
    html.Div(id='user-det', children=[
        html.Img(id='pic', style={'height': '150px', 'width': '150px', 'borderRadius': '5%', 'marginRight': '-50%'}),
        html.Div(id='user', style={'height': '150px', 'width': '150px'}),
        dcc.Input(id='input-box', value='1', type='text', style={'borderRadius': '5px'}),
        html.Div(id='tips', style={'height': '100px', 'width': '300px', 'boxShadow': '1px 1px 1px 1px Gray', 'borderRadius': '5px'})
    ], style={'marginLeft': '5%', 'columnCount': 3}
    ),

    #plotted graphs
    html.Div(id='inter', children=[
        dcc.Graph(id='pers_det'),
        dcc.Graph(id='performance')
    ], style={'columnCount': 2}),

    dcc.Graph(id='agg_det'),

    #recommendations
    html.H6('Recommendations', style={'color': 'SteelBlue', 'marginLeft': '2%'}),
    html.Div(id='recom', children=[
        dcc.Checklist(id='opts')
    ])
], style=styles['main'])

@app.callback(
    Output(component_id='user', component_property='children'),
    [Input(component_id='input-box', component_property='value')]
)
def user_name(user_num):
    perfom_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    details = "Name: User#{}\nLocation: Wells Fargo\nCategory:\n{}".format(user_num, perfom_dict[2])
    return details

@app.callback(
    Output(component_id='pic', component_property='src'),
    [Input(component_id='input-box', component_property='value')]
)
def display_image(user_id):
    range_num = int(user_id) % 57
    print(range_num)
    return encode_image('diverseui-10-06/image-{}.png'.format(str(range_num)))

@app.callback(
    Output(component_id='pers_det', component_property='figure'),
    [Input(component_id='input-box', component_property='value')]
)
def personal_visual(user_id):
    dat = data.copy()
    dat.columns = ['heating','bath','kitchen','TV','transportation','waste']
    df_pi = pd.DataFrame(index=data.index)
    df_pi['waste'] = dat['waste']
    df_pi['transportation'] = dat['transportation']
    df_pi['other'] = dat['heating'] + dat['bath'] + dat['kitchen'] + dat['TV']
    user = df_pi.iloc[int(user_id), :]
    labels = df_pi.columns
    values = user.values
    colors = ['#6B8E23', '#9ACD32','#98FB98']
    trace = go.Pie(labels=labels, values=values,
                hoverinfo='label+percent', textinfo='value',
                hole=.3,
                textfont=dict(size=15),
                marker=dict(colors=colors,
                line=dict(color='#000000', width=0.5)))
    return {
        'data': [trace],
        'layout': go.Layout(
            hovermode='closest'
        )
    }

@app.callback(
    Output(component_id='agg_det', component_property='figure'),
    [Input(component_id='input-box', component_property='value')]
)
def aggregate_visual(user_id):
    dat = data.copy()
    df_comp = pd.DataFrame()
    df_comp['summation'] = dat.sum(axis=1)
    user = df_comp.iloc[int(user_id),:]

    trace = go.Histogram(
        x = df_comp['summation'],
        opacity=0.5,
        marker=dict(
            color='#9ACD32'
        ),
        name='Population'
    )
    trace2 = go.Scatter(
        x=[float(user),float(user)],
        y=[0,110],
        text = 'You',
        hoverinfo = 'text',
        mode = 'lines',
        marker=dict(
            color='#6B8E23'
        ),
        name='Where you stand'
    )
    return {
        'data': [trace,trace2],
        'layout': go.Layout(
            hovermode='closest'
        )
    }

@app.callback(
    Output(component_id='performance', component_property='figure'),
    [Input(component_id='input-box', component_property='value')]
)
def performance_visual(user_id):
    dat = data.copy()
    scaler = MinMaxScaler()
    dat['Group1'] = 1 - scaler.fit_transform(dat['Group1'].values.reshape(-1,1))
    dat['Group2'] = 1 - scaler.fit_transform(dat['Group2'].values.reshape(-1,1))
    dat['Group3'] = 1 - scaler.fit_transform(dat['Group3'].values.reshape(-1,1))
    dat['Group4'] = 1 - scaler.fit_transform(dat['Group4'].values.reshape(-1,1))
    dat['Group5'] = 1 - scaler.fit_transform(dat['Group5'].values.reshape(-1,1))
    dat['Group6'] = 1 - scaler.fit_transform(dat['Group6'].values.reshape(-1,1))

    user = dat.iloc[int(user_id), :]
    mean = dat.mean()
    scatter1 = go.Scatterpolar(
        r = user,
        theta = ['Heating','Bath','Kitchen', 'TV', 'Transportation', 'Waste'],
        fill = 'toself',
        name = 'Current'
    )
    scatter2 = go.Scatterpolar(
        r = mean,
        theta = ['Heating','Bath','Kitchen', 'TV', 'Transportation', 'Waste'],
        fill = 'toself',
        name = 'Change'
    )
    layout = go.Layout(
        polar = dict(
            radialaxis = dict(
            visible = True,
            range = [0, 1]
            )
        )
    )
    return {
        'data': [scatter1, scatter2],
        'layout': layout
    }

# @app.callback(
#     Output(component_id='opts', component_property='options'),
#     [Input(component_id='input-box', component_property='value')]
# )
# def recommend(user_val):
#     options = []
#     vals = [1, 4]
#     #for i in vals:
#     row = products[1]
#     d = {'p1': row[0],
#          'p2': row[1],
#          'pr1': row[2],
#          'pr2': row[3],
#          'l1': row[4],
#          'l2': row[5]}
#     #options.append(d)
#     return d

def tip_show(user_val):
    pass

if __name__ == '__main__':
    app.run_server(debug=True)
