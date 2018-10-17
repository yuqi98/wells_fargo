from sklearn.metrics import calinski_harabaz_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits import mplot3d
from sklearn.mixture import GaussianMixture
from scipy.stats import chisquare
from scipy import stats
from sklearn.decomposition import PCA
import plotly.plotly as py
import itertools
import plotly.graph_objs as go
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings('ignore')

# Read data
data = pd.read_csv('first_preprocess.csv',index_col='user_num')
data.columns = ['heating','bath','kitchen','TV','transportation','waste']
clustering = SpectralClustering(n_clusters=4,
    assign_labels="discretize",
    random_state=0).fit(data)
data['clusters'] = clustering.labels_+1

df = pd.read_csv('fillna_lifequality.csv')
dd = pd.read_csv('carbon.csv',index_col=0, na_values=['(NA)']).fillna(0)


id_num = 973
group_num = data.loc[id_num]["clusters"]
t = id_num-1;
cur_df = df.iloc[t*27:t*27+27,:]
new = cur_df.iloc[:,6:16]
new = new.reset_index()
new=new.drop(columns=["index"])
ddd = dd.iloc[:,2:12]
ddd = ddd.reset_index()
ddd = ddd.drop(columns=["index"])

cols = df["Activity"].unique()
result = pd.DataFrame(new.values*ddd.values, index=cols)
result = result.sum(axis=1)

if group_num == 1:
    heat_h = cur_df.loc[cur_df.Activity=="shower - short", "Quality_of_Life_Importance__1_10"].values
    heat_l = cur_df.loc[cur_df.Activity=="shower - long (> 3 min)", "Quality_of_Life_Importance__1_10"].values
    diff1 = int(heat_h - heat_l)
    #print(diff1)
    
    diff2 = float(result.loc["shower - short"] - result.loc["shower - long (> 3 min)"])
    
    cons_h = cur_df.loc[cur_df.Activity=="shower - short", "Consumption"].values
    cons_l = cur_df.loc[cur_df.Activity=="shower - long (> 3 min)", "Consumption"].values
    
    #print(int(cons_l))
    
    if (diff1>0 and cons_l!=0):
        print('Change your long shower to short shower by {} times'.format(int(cons_l)))
        if diff2>0:
            print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2)))
        else:
            print('Your carbon footprint will be successfully reduced')
        print('And your life quality will be increased by {} percentage'.format(int(100*float(diff1/heat_l))))
        
    else: 
        if (cons_l != 0):
            print('Change your long shower to short shower by {} times'.format(int(cons_l/2)))
            if diff2>0:
                print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2/2)))
            else:
                print('Your carbon footprint will be successfully reduced')


if group_num == 2:
    heat_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Quality_of_Life_Importance__1_10"].values
    heat_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Quality_of_Life_Importance__1_10"].values
    diff1 = int(heat_h - heat_l)
    #print(diff1)
    
    diff2 = float(result.loc["Household heating < 70F"] - result.loc["Household heating => 70F"])
    
    cons_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Consumption"].values
    cons_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Consumption"].values
    
    #print(int(cons_l))
    
    if (diff1>0 and cons_l!=0):
        print('Change your heating time with tempreture <70F to >=70F by {} hours'.format(int(cons_l)))
        if diff2>0:
            print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2)))
        else:
            print('Your carbon footprint will be successfully reduced')
        print('And your life quality will be increased by {} percentage'.format(int(100*float(diff1/heat_l))))
        
    else: 
        if (cons_l != 0):
            print('Change your heating time with tempreture <70F to >=70F by {} hours'.format(int(cons_l/2)))
            if diff2>0:
                print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2/2)))
            else:
                print('Your carbon footprint will be successfully reduced')
        else:
            cons = cur_df.loc[cur_df.Activity=="Use of air conditioner", "Consumption"].values
            cost = result.loc["Use of air conditioner"]
            print('Reduce your use of air conditioner by {} hours'.format(float(re)))
            print('Your carbon footprint will be reduced by {}'.format(float(re*cost)))
    
    heat_h = cur_df.loc[cur_df.Activity=="car trips - 2+ people with multiple end points", "Quality_of_Life_Importance__1_10"].values
    heat_l = cur_df.loc[cur_df.Activity=="car trips- self only", "Quality_of_Life_Importance__1_10"].values
    diff1 = int(heat_h - heat_l)
    #print(diff1)
    
    diff2 = float(result.loc["car trips- self only"] - result.loc["car trips - 2+ people with multiple end points"])
    
    cons_h = cur_df.loc[cur_df.Activity=="car trips - 2+ people with multiple end points", "Consumption"].values
    cons_l = cur_df.loc[cur_df.Activity=="car trips- self only", "Consumption"].values
    
    #print(int(cons_l))
    
    if (diff1>0 and cons_l!=0):
        print('Adding number of people of your car trips to more than 2'.format(int(cons_l)))
        if diff2>0:
            print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2)))
        else:
            print('Your carbon footprint will be successfully reduced')
        print('And your life quality will be increased by {} percentage'.format(int(100*float(diff1/heat_l))))
        
    else: 
        if (cons_l != 0):
            print('Adding number of people of your car trips to more than 2'.format(int(cons_l/2)))
            if diff2>0:
                print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2/2)))
            else:
                print('Your carbon footprint will be successfully reduced')



if group_num == 3:
    
    diff2 = float(0.0419*2)
    
    cons_l = cur_df.loc[cur_df.Activity=="bags of garbage disposed", "Consumption"].values
    
    #print(int(cons_l))
    
    if (cons_l!=0):
        print('By recyclying your garbage')
        if diff2>0:
            print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2)))
        else:
            print('Your carbon footprint will be successfully reduced')

if group_num == 4:
    heat_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Quality_of_Life_Importance__1_10"].values
    heat_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Quality_of_Life_Importance__1_10"].values
    diff1 = int(heat_h - heat_l)
    #print(diff1)
    
    diff2 = float(result.loc["Household heating < 70F"] - result.loc["Household heating => 70F"])
    
    cons_h = cur_df.loc[cur_df.Activity=="Household heating => 70F", "Consumption"].values
    cons_l = cur_df.loc[cur_df.Activity=="Household heating < 70F", "Consumption"].values
    
    #print(int(cons_l))
    
    if (diff1>0 and cons_l!=0):
        print('Change your heating time with tempreture <70F to >=70F by {} hours'.format(int(cons_l)))
        if diff2>0:
            print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2)))
        else:
            print('Your carbon footprint will be successfully reduced')
        print('And your life quality will be increased by {} percentage'.format(int(100*float(diff1/heat_l))))
        
    else: 
        if (cons_l != 0):
            print('Change your heating time with tempreture <70F to >=70F by {} hours'.format(int(cons_l/2)))
            if diff2>0:
                print('Your carbon footprint will be reduced by {}'.format(float(cons_l*diff2/2)))
            else:
                print('Your carbon footprint will be successfully reduced')
        else:
            cons = cur_df.loc[cur_df.Activity=="Use of air conditioner", "Consumption"].values
            cost = result.loc["Use of air conditioner"]
            print('Reduce your use of air conditioner by {} hours'.format(float(cons/2)))
            print('Your carbon footprint will be reduced by {}'.format(float((cons/2)*cost)))








