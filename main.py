from flask import Flask, render_template, request, redirect, Response, jsonify, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import csv
 
app = Flask(__name__)

# @app.route('/', methods=['POST'])
# def home():

@app.route('/pca', methods=['GET', 'POST'])
def pca(components=6):


    # scales the data
    scaled_data,attributes = data_scaling() 

    if request.method == "GET":

        ### PCA  ###

        pca_data, pca_components, explained_variance_ratio, loadings, eigen_values, cumulative_sum, loadings_biplot = calc_pca(components,scaled_data)

        ### Scree plot ###
        PC_values = list(np.arange(pca_components) + 1)

        loadings = loadings.to_dict("records")
        attributes = list(map(str, attributes))

        df = pd.read_csv("Data/data.csv")

        # ENCODING
        label_encoder = preprocessing.LabelEncoder()
        df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

        df.drop(columns=['id'], inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.loc[:, [
        "radius_mean",
        "perimeter_worst",
        "perimeter_mean",
        "perimeter_se",
        "radius_se",
        "compactness_worst",
        "concavity_se",
        "concave points_mean",
        "texture_mean",
        "symmetry_worst",
        "diagnosis"
        ]]

        data_list = df.to_dict(orient='records')
        print("data list:",len(data_list))


        data = {
            "pc_values":list(map(int, PC_values)),
            "variance":list(map(float, explained_variance_ratio)),
            "components":components,
            "loadings":loadings,
            "eigen_values":list(map(int, eigen_values)),
            "flag":False,
            "attributes": attributes,
            "data_list": data_list
        }

        return render_template("index.html", data=data)

@app.route('/biplot', methods=['GET', 'POST'])
def biplot():
        # Load data
        df = pd.read_csv("Data/data.csv")

        # ENCODING
        label_encoder = preprocessing.LabelEncoder()
        df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

        df.drop(columns=['id'], inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.loc[:, [
        "radius_mean",
        "perimeter_worst",
        "perimeter_mean",
        "perimeter_se",
        "radius_se",
        "compactness_worst",
        "concavity_se",
        "concave points_mean",
        "texture_mean",
        "symmetry_worst"
         ]]
        columns = list(df.columns) #

        # SCALING
        scaler = MinMaxScaler()
        scaler.fit(df)

        scaled_data = scaler.transform(df)

        # PCA
        components = 2
        pca = PCA(n_components=components)
        pca.fit(scaled_data)

        pca_data = pca.transform(scaled_data) #
        pca_components = pca.n_components_
        explained_variance_ratio = pca.explained_variance_ratio_
        loadings = pca.components_ #  

        # Convert data to JSON format
        pca_data = pca_data.tolist() 
        loadings = loadings.tolist()
        evr = explained_variance_ratio.tolist()

        paired_loadings = list(zip(loadings[0], loadings[1]))
        # print(loadings)
        # print(paired_loadings)

        data = {
            "pca_data": pca_data,
            "evr":evr,
            "columns":columns,
            "loadings": paired_loadings,
        }
        # Render template with data
        return render_template('biplot.html', data=data)


@app.route('/kmeans')
def kmeans():
    return render_template('kmeans.html')


    return render_template('scatterplot.html', data=data)
### HELPER FUNCTIONS ###

def data_scaling():
    df = pd.read_csv("Data/data.csv")

    ### Encoding the labels  ###
    label_encoder = preprocessing.LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

    df.drop(columns=['id'], inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # print("df=",df)

    normal_data = df

    attributes = df.columns


    ### Scaling the numerical data  ###
    scaler = MinMaxScaler()
    scaler.fit(df)

    scaled_data = scaler.transform(df)
    print("scaled data",scaled_data)

    return scaled_data, attributes

def calc_pca(components, data):

    pca = PCA(n_components=components)
    pca.fit(data)
    pca_data = pca.transform(data)

    pca_components = pca.n_components_
    explained_variance_ratio = pca.explained_variance_ratio_

    loadings = pd.DataFrame(pca.components_.T, columns=['PC'+str(i) for i in range(1,components+1)])

    loadings_biplot = pca.components_
    # print(pca.components_.T.shape)

    eigen_values = list(pca.singular_values_)
    # print("eigen values:",type(eigen_values))
    cumulative_sum= np.cumsum(explained_variance_ratio)

    return pca_data, pca_components, explained_variance_ratio, loadings, eigen_values, cumulative_sum, loadings_biplot


# @app.route('/data')
# def data():
#     with open('Data/data.csv', newline='') as csvfile:
#         data = list(csv.DictReader(csvfile))
#     return Response(
#         csv.dumps(data),
#         mimetype="text/csv",
#         headers={"Content-disposition":
#                  "attachment; filename=data.csv"})

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)