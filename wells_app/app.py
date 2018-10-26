import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import plotly.plotly as py
import plotly.graph_objs as go

from flask import send_file
import base64
import datetime
import io

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering


#css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
styles = {
    'main': {'borderRadius': '2px', 'padding': '5px 5px 5px 5px',
              'marginLeft': 'auto', 'marginRight': 'auto', "width": "98%"},
    'sub-menu': {'color': '#FFC002', 'marginLeft': '4%'},
    'sub-sub': {'marginLeft': '5%', 'borderRadius': '2px', 'padding': '1%',
                'boxShadow': '0px 0px 2px grey'}
}

#data
data = pd.read_csv('./data/first_preprocess_1.csv', index_col='user_num')
carbon_p = pd.read_csv('./data/carbon_footprint.csv',na_values=['(NA)']).fillna(0)
products = pd.read_csv('./data/products.csv', index_col='group')
products = products.iloc[:, ~products.columns.str.contains('^Unnamed')]
df = pd.read_csv('./data/fillna_lifequality.csv')
dd = pd.read_csv('./data/carbon.csv',index_col=0, na_values=['(NA)']).fillna(0)

#grouping
grou = data.copy()
grou.columns = ['heating','bath','kitchen','TV','transportation','waste']
df_pi = pd.DataFrame(index=data.index)
df_pi['waste'] = grou['waste']
df_pi['transportation'] = grou['transportation']
df_pi['other'] = grou['heating'] + grou['bath'] + grou['kitchen'] + grou['TV']

#summation
df_comp = pd.DataFrame()
df_comp['summation'] = data.sum(axis=1)

#MinMaxScaler
mms = data.copy()
scaler = MinMaxScaler()
mms['Group1'] = 1 - scaler.fit_transform(mms['Group1'].values.reshape(-1,1))
mms['Group2'] = 1 - scaler.fit_transform(mms['Group2'].values.reshape(-1,1))
mms['Group3'] = 1 - scaler.fit_transform(mms['Group3'].values.reshape(-1,1))
mms['Group4'] = 1 - scaler.fit_transform(mms['Group4'].values.reshape(-1,1))
mms['Group5'] = 1 - scaler.fit_transform(mms['Group5'].values.reshape(-1,1))
mms['Group6'] = 1 - scaler.fit_transform(mms['Group6'].values.reshape(-1,1))

#cluster
clus = data.copy()
clus.columns = ['heating','bath','kitchen','TV','transportation','waste']
clustering = SpectralClustering(n_clusters=4,
    assign_labels="discretize",
    random_state=0).fit(clus)
clus['clusters'] = clustering.labels_+1

app = dash.Dash("Wells-Carbo", external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions']=True

app.layout = html.Div([
    html.Div([
        html.H1('Wells', style={'textAlign': 'right', 'color': '#CA0009'}),
        html.H1(' - ', style={'textAlign': 'center', 'color': 'black'}),
        html.H1('Carbo', style={'textAlign': 'left', 'color': '#FFC002'})
    ], style={'columnCount': 3}),

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Explore Data', value='tab-1'),
        dcc.Tab(label='New User', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
                    html.P('Enter UserID[1 - 1002]'),
                    dcc.Input(id='input-box', value='1', type='text', style={'borderRadius': '5px'}),
                    #plotted graphs
                    dcc.Graph(id='agg_det'),
                    html.Div(id='inter', children=[
                        dcc.Graph(id='pers_det'),
                        dcc.Graph(id='performance')
                    ], style={'columnCount': 2}),

                    #recommendations
                    html.H5('Recommendations', style={'color': '#CA0009', 'marginLeft': '2%'}),
                    html.Div([
                        html.H6('Products', style=styles['sub-menu']),
                        html.Div(id='prods', style=styles['sub-sub']),
                        html.H6('Tips', style=styles['sub-menu']),
                        html.Div(id='recoms',  style=styles['sub-sub'])
                ])
        ], style=styles['main'])
    elif tab == 'tab-2':
        return html.Div([
                    html.Div([
dcc.Markdown('''
    Follow Procedure Below to check your carbon_footprint
    And possible improvements you can make

    *Click on the Download File Button
    *Fill in your details
        *Consumption column is in count of units columns
        *Quality_of_Life_Importance__1_10 column is range of 100%
        *For Type of Energy Selection:
            *Input 1 for Yes
            *Input 0 for No
    *Drag and Drop or Select File to Upload File
    ***Please note, your data will be added to database to improve results
'''),

                        html.A(
                            html.Button('Download File'),
                            id='download',
                            href="/dash/urlToDownload"
                        ),
                    ], style={'margin': '0px 0px 0px 30px'}),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select File')
                        ]),
                        style={
                            'width': '95%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '0px 0px 0px 30px'
                        },
                    ),
                    html.Div(id='show-data')
                ])

@app.server.route('/dash/urlToDownload')
def download_csv():
    return send_file('./data/input_details.csv',
                     mimetype='text/csv',
                     attachment_filename='downloadFile.csv',
                     as_attachment=True)

@app.callback(
    Output(component_id='pers_det', component_property='figure'),
    [Input(component_id='input-box', component_property='value')]
)
def personal_visual(user_id):
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
            hovermode='closest',
            xaxis= dict(title='Cummulative Carbon Footprint'),
            yaxis= dict(title='Population')
        )
    }

@app.callback(
    Output(component_id='performance', component_property='figure'),
    [Input(component_id='input-box', component_property='value')]
)
def performance_visual(user_id):
    user = mms.iloc[int(user_id), :]
    mean = mms.mean()
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

@app.callback(
    Output(component_id='recoms', component_property='children'),
    [Input(component_id='input-box', component_property='value')]
)
def recom_and_reduce(user_num):
    id_num = int(user_num)
    group_num = clus.loc[id_num]["clusters"]
    t = id_num-1;
    cur_df = df.iloc[t*27:t*27+27,:]
    new = cur_df.iloc[:,6:16]
    new = new.reset_index()
    new = new.iloc[:, 1:]
    ddd = dd.iloc[:,2:12]
    ddd = ddd.reset_index()
    ddd = ddd.iloc[:, 1:]
    cols = df["Activity"].unique()
    result = pd.DataFrame(new.values*ddd.values, index=cols)
    result = result.sum(axis=1)

    def best_rec(high,low,diff,all_carbon,cons_h,cons_l):
        per_quality = abs((high-low)/low)
        per_carbonchange = abs(diff/all_carbon)
        proportion = per_carbonchange / (per_quality+per_carbonchange)
        return (cons_l*proportion)

    recom_keeper = []
    if group_num == 1:
        heat_h = cur_df.loc[cur_df.Activity=="shower - short", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="shower - long (> 3 min)", "Quality_of_Life_Importance__1_10"].values
        diff1 = abs(int(heat_h - heat_l))
        diff2 = float(result.loc["shower - short"] - result.loc["shower - long (> 3 min)"])
        cons_h = cur_df.loc[cur_df.Activity=="shower - short", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="shower - long (> 3 min)", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Change your long shower to short shower by {} times.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))
        else:
            if (cons_l != 0 and diff2 > 0):
                carbon = result['shower - short'] + result['shower - long (> 3 min)']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Change your long shower to short shower by {} times.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(best_change*diff2))))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Change your long shower to short shower by {} times.'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
    if group_num == 2:
        heat_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Quality_of_Life_Importance__1_10"].values
        diff1 = int(heat_h - heat_l)
        diff2 = float(result.loc["Household heating < 70F"] - result.loc["Household heating => 70F"])

        cons_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))

        else:
            if (cons_l != 0 and diff2 > 0):
                carbon = result['Household heating => 70F'] + result['Household heating < 70F']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(best_change*diff2))))
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Change your heating time with tempreture >=70F to <70F by {} hours.'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
                else:
                    cons = cur_df.loc[cur_df.Activity=="Use of air conditioner", "Consumption"].values
                    cost = result.loc["Use of air conditioner"]
                    recom_keeper.append(html.P('Reduce your use of air conditioner by {} hours.'.format(float(cons))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons*cost))))

        heat_h = cur_df.loc[cur_df.Activity=="car trips - 2+ people with multiple end points", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="car trips- self only", "Quality_of_Life_Importance__1_10"].values
        diff1 = int(heat_h - heat_l)
        diff2 = float(result.loc["car trips- self only"] - result.loc["car trips - 2+ people with multiple end points"])
        cons_h = cur_df.loc[cur_df.Activity=="car trips - 2+ people with multiple end points", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="car trips- self only", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Adding number of people of your car trips to more than 2.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))
        else:
            if (cons_l != 0 and diff2 > 0):
                carbon = result['car trips - 2+ people with multiple end points'] + result['car trips- self only']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Adding number of people of your car trips to more than 2 for {} trips.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(best_change*diff2))))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Do self driving car trips for {} trips'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
    if group_num == 3:
        diff2 = float(0.0419*2)
        cons_l = cur_df.loc[cur_df.Activity=="bags of garbage disposed", "Consumption"].values
        if (cons_l!=0):
            recom_keeper.append(html.P('By recyclying your garbage.'))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
    if group_num == 4:
        heat_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Quality_of_Life_Importance__1_10"].values
        diff1 = int(heat_h - heat_l)
        diff2 = float(result.loc["Household heating < 70F"] - result.loc["Household heating => 70F"])
        cons_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))
        else:
            if (cons_l != 0 and diff2>0):
                carbon = result['Household heating => 70F'] + result['Household heating < 70F']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(diff2*best_change))))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Change your heating time with tempreture >=70F to <70F by {} hours.'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
                else:
                    cons = cur_df.loc[cur_df.Activity=="Use of air conditioner", "Consumption"].values
                    cost = result.loc["Use of air conditioner"]
                    recom_keeper.append(html.P('Reduce your use of air conditioner by {} hours.'.format(float(cons/2))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float((cons/2)*cost))))
    return html.Div(children=recom_keeper)

@app.callback(
    Output(component_id='prods', component_property='children'),
    [Input(component_id='input-box', component_property='value')]
)
def suggest_products(user_num):
    id_num = int(user_num)
    group_num = int(clus.loc[id_num]["clusters"])
    val = products.iloc[group_num].values
    return html.Div([
            dcc.Checklist(
                options=[
                    {'label': '{} {}'.format(val[0], val[2]), 'value': 'prod1'},
                    {'label': '{} {}'.format(val[1], val[3]), 'value': 'prod2'}
                ],
                values=["prod1"]
            ),
            html.Div([
                html.A("Product 1", href=val[4], target="_blank"),
                html.Br(),
                html.A("Product 2", href=val[5], target="_blank")
            ])

        ], style={'columnCount': 2})

@app.callback(Output('show-data', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(contents, names, dates):
    if contents is not None:
        children = [parse_contents(contents, names, dates)]
        return children

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            later = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            later = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    df['waste_management'] = 0.0
    for x in ['bags','hazardous','large']:
        condition = df['Activity'].str.match(x) & df['Consumption']>0.0
        df.loc[condition, 'waste_management'] = 1.0
    df2 = carbon_p[['solar powered  water heater', 'gas water heater',
           'electric water heater - peak hours',
           'electric water heater - off peak hours', 'gas', 'natural gas','hybrid','electric - peak hours',
           'electric - off peak hours',
           'Jet Fuel','waste management']]

    df_part1 = df.iloc[:,5:].copy()
    df_part2 = df2
    data_cf = pd.DataFrame(df_part1.values*df_part2.values, columns=df_part1.columns, index=df_part1.index)

    data_complete = pd.concat([df.iloc[:,:5],data_cf], axis=1)
    data_complete['Consumption'] = data_complete['Consumption'].abs()
    data_complete['n_utils'] = data_complete.iloc[:, -11:].astype(bool).sum(axis=1)
    data_complete['sum_cf'] = data_complete.loc[:,'solar_powered__water_heater':'waste_management'].sum(axis = 1)
    data_complete['cf'] = ((data_complete['sum_cf']*data_complete['Consumption'])/data_complete['n_utils'])
    data_complete['cf'].fillna(0, inplace=True)

    new_data = data_complete.groupby('Group').mean()
    list_cf = list(new_data['cf'])

    df_final = pd.DataFrame(columns = ['Group1','Group2','Group3','Group4','Group5','Group6'])
    df_final.loc[0] = list_cf

    group = df_final.copy()
    group.columns = ['heating','bath','kitchen','TV','transportation','waste']
    df_p = pd.DataFrame(index=group.index)
    df_p['waste'] = group['waste']
    df_p['transportation'] = group['transportation']
    df_p['other'] = group['heating'] + group['bath'] + group['kitchen'] + group['TV']

    nana = df_p.iloc[0, :]
    labels = df_p.columns
    values = nana.values
    colors = ['#6B8E23', '#9ACD32','#98FB98']
    trace1 = go.Pie(labels=labels, values=values,
                hoverinfo='label+percent', textinfo='value',
                hole=.3,
                textfont=dict(size=15),
                marker=dict(colors=colors,
                line=dict(color='#000000', width=0.5)))

    big = data.copy()
    big = pd.concat([big, df_final], ignore_index=True)
    scaler = MinMaxScaler()
    big['Group1'] = 1 - scaler.fit_transform(big['Group1'].values.reshape(-1,1))
    big['Group2'] = 1 - scaler.fit_transform(big['Group2'].values.reshape(-1,1))
    big['Group3'] = 1 - scaler.fit_transform(big['Group3'].values.reshape(-1,1))
    big['Group4'] = 1 - scaler.fit_transform(big['Group4'].values.reshape(-1,1))
    big['Group5'] = 1 - scaler.fit_transform(big['Group5'].values.reshape(-1,1))
    big['Group6'] = 1 - scaler.fit_transform(big['Group6'].values.reshape(-1,1))
    import random
    rand = random.randint(1, 1000)
    user_min_max = big.iloc[rand, :]
    mean = big.mean()
    scatter1 = go.Scatterpolar(
        r = user_min_max,
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

    temp = df_final.copy()
    temp['summation'] = data.sum(axis=1)
    user_sum = df_comp.iloc[0,:]
    trace2 = go.Histogram(
        x = df_comp['summation'],
        opacity=0.5,
        marker=dict(
            color='#9ACD32'
        ),
        name='Population'
    )
    trace3 = go.Scatter(
        x=[float(user_sum),float(user_sum)],
        y=[0,110],
        text = 'You',
        hoverinfo = 'text',
        mode = 'lines',
        marker=dict(
            color='#6B8E23'
        ),
        name='Where you stand'
    )

    clust = data.copy()
    clust = pd.concat([big, df_final], ignore_index=True)
    #clust.to_csv('./data/first_preprocess.csv')
    clust.columns = ['heating','bath','kitchen','TV','transportation','waste']
    clustering = SpectralClustering(n_clusters=4,
        assign_labels="discretize",
        random_state=0).fit(clust)
    clust['clusters'] = clustering.labels_+1

    id_num = clust.shape[0] - 1
    group_num = clust.loc[id_num]["clusters"]
    t = 50-1;
    dfp = pd.read_csv('./data/fillna_lifequality.csv')
    #dfp = pd.concat([dfp, later.iloc[:, :27]], ignore_index=True)
    cur_df = dfp.iloc[t*27:t*27+27,:]
    new = cur_df.iloc[:,6:16]
    new = new.reset_index()
    new = new.iloc[:, 1:]
    ddd = dd.iloc[:,2:12]
    ddd = ddd.reset_index()
    ddd = ddd.iloc[:, 1:]
    cols = dfp["Activity"].unique()
    result = pd.DataFrame(new.values*ddd.values, index=cols)
    result = result.sum(axis=1)

    def best_rec(high,low,diff,all_carbon,cons_h,cons_l):
        per_quality = abs((high-low)/low)
        per_carbonchange = abs(diff/all_carbon)
        proportion = per_carbonchange / (per_quality+per_carbonchange)
        return (cons_l*proportion)

    recom_keeper = []
    if group_num == 1:
        heat_h = cur_df.loc[cur_df.Activity=="shower - short", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="shower - long (> 3 min)", "Quality_of_Life_Importance__1_10"].values
        diff1 = int(heat_h - heat_l)
        diff2 = float(result.loc["shower - short"] - result.loc["shower - long (> 3 min)"])
        cons_h = cur_df.loc[cur_df.Activity=="shower - short", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="shower - long (> 3 min)", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Change your long shower to short shower by {} times.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))
        else:
            if (cons_l != 0 and diff2 > 0):
                carbon = result['shower - short'] + result['shower - long (> 3 min)']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Change your long shower to short shower by {} times.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(best_change*diff2))))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Change your long shower to short shower by {} times.'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
    if group_num == 2:
        heat_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Quality_of_Life_Importance__1_10"].values
        diff1 = int(heat_h - heat_l)
        diff2 = float(result.loc["Household heating < 70F"] - result.loc["Household heating => 70F"])

        cons_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))

        else:
            if (cons_l != 0 and diff2 > 0):
                carbon = result['Household heating => 70F'] + result['Household heating < 70F']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(best_change*diff2))))
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Change your heating time with tempreture >=70F to <70F by {} hours.'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
                else:
                    cons = cur_df.loc[cur_df.Activity=="Use of air conditioner", "Consumption"].values
                    cost = result.loc["Use of air conditioner"]
                    recom_keeper.append(html.P('Reduce your use of air conditioner by {} hours.'.format(float(cons))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons*cost))))

        heat_h = cur_df.loc[cur_df.Activity=="car trips - 2+ people with multiple end points", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="car trips- self only", "Quality_of_Life_Importance__1_10"].values
        diff1 = int(heat_h - heat_l)
        diff2 = float(result.loc["car trips- self only"] - result.loc["car trips - 2+ people with multiple end points"])
        cons_h = cur_df.loc[cur_df.Activity=="car trips - 2+ people with multiple end points", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="car trips- self only", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Adding number of people of your car trips to more than 2.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))
        else:
            if (cons_l != 0 and diff2 > 0):
                carbon = result['car trips - 2+ people with multiple end points'] + result['car trips- self only']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Adding number of people of your car trips to more than 2 for {} trips.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(best_change*diff2))))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Do self driving car trips for {} trips'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
    if group_num == 3:
        diff2 = float(0.0419*2)
        cons_l = cur_df.loc[cur_df.Activity=="bags of garbage disposed", "Consumption"].values
        if (cons_l!=0):
            recom_keeper.append(html.P('By recyclying your garbage.'))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced.'))
    if group_num == 4:
        heat_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Quality_of_Life_Importance__1_10"].values
        heat_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Quality_of_Life_Importance__1_10"].values
        diff1 = int(heat_h - heat_l)
        diff2 = float(result.loc["Household heating < 70F"] - result.loc["Household heating => 70F"])
        cons_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Consumption"].values
        cons_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Consumption"].values
        if (diff1>0 and cons_l!=0):
            recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(cons_l))))
            if diff2>0:
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_l*diff2))))
            else:
                recom_keeper.append(html.P('Your carbon footprint will be successfully reduced'))
            recom_keeper.append(html.P('And your life quality will be increased by {} percentage.'.format(int(100*float(diff1/heat_l)))))
        else:
            if (cons_l != 0 and diff2>0):
                carbon = result['Household heating => 70F'] + result['Household heating < 70F']
                best_change = best_rec(heat_h,heat_l,diff2,carbon,cons_h,cons_l)
                recom_keeper.append(html.P('Change your heating time with tempreture <70F to >=70F by {} hours.'.format(int(best_change))))
                recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(diff2*best_change))))
            else:
                if (cons_l != 0 and diff2 <= 0):
                    recom_keeper.append(html.P('Change your heating time with tempreture >=70F to <70F by {} hours.'.format(int(cons_h))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float(cons_h*(-diff2)))))
                else:
                    cons = cur_df.loc[cur_df.Activity=="Use of air conditioner", "Consumption"].values
                    cost = result.loc["Use of air conditioner"]
                    recom_keeper.append(html.P('Reduce your use of air conditioner by {} hours.'.format(float(cons/2))))
                    recom_keeper.append(html.P('Your carbon footprint will be reduced by {}.'.format(float((cons/2)*cost))))

    group_num = int(clust.loc[id_num]["clusters"])
    val = products.iloc[group_num].values
    return html.Div([
                html.H5('Below are your results', style={'color': '#CA0009', 'marginLeft': '2%'}),
                #plotted graphs
                dcc.Graph(
                    figure={
                        'data': [trace2, trace3],
                        'layout': go.Layout(
                            hovermode='closest',
                            xaxis= dict(title='Cummulative Carbon Footprint'),
                            yaxis= dict(title='Population')
                        )
                    }
                ),
                html.Div(id='inter', children=[
                    dcc.Graph(id='pers_det1', figure={
                        'data': [trace1],
                        'layout': go.Layout(
                            hovermode='closest'
                        )
                    }),
                    dcc.Graph(id='performance1', figure={
                        'data': [scatter1, scatter2],
                        'layout': layout
                    })
                ], style={'columnCount': 2}),

                #recommendations
                html.H5('Recommendations', style={'color': '#CA0009', 'marginLeft': '2%'}),
                html.Div([
                    html.H6('Products', style=styles['sub-menu']),
                    html.Div([html.Div([
                            dcc.Checklist(
                                options=[
                                    {'label': '{} {}'.format(val[0], val[2]), 'value': 'prod1'},
                                    {'label': '{} {}'.format(val[1], val[3]), 'value': 'prod2'}
                                ],
                                values=["prod1"]
                            ),
                            html.Div([
                                html.A("Product 1", href=val[4]),
                                html.Br(),
                                html.A("Product 2", href=val[5])
                            ])
                        ], style={'columnCount': 2})], style=styles['sub-sub'])
                    ]),
                    html.H6('Tips', style=styles['sub-menu']),
                    html.Div(html.Div(children=recom_keeper),  style=styles['sub-sub'])
        ], style=styles['main'])



if __name__ == '__main__':
    app.run_server()
