import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import pandas as pd
import requests

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='input1', type='number', placeholder='OverallQual'),
    dcc.Input(id='input2', type='number', placeholder='GrLivArea'),
    dcc.Input(id='input3', type='number', placeholder='GarageCars'),
    dcc.Input(id='input4', type='number', placeholder='GarageArea'),
    dcc.Input(id='input5', type='number', placeholder='TotalBsmtSF'),
    dcc.Input(id='input6', type='number', placeholder='FullBath'),
    dcc.Input(id='input7', type='number', placeholder='YearBuilt'),
    html.Button('Predict', id='predict-button'),
    html.Div(id='prediction-output')
])

@app.callback(
    dd.Output('prediction-output', 'children'),
    [dd.Input('predict-button', 'n_clicks')],
    [dd.State(f'input{i}', 'value') for i in range(1, 8)]
)
def update_output(n_clicks, *values):
    if n_clicks is not None:
        features = {f'{key}': [val] for key, val in zip(['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt'], values)}
        response = requests.post('http://127.0.0.1:5000/predict', json=features)
        prediction = response.json()['prediction']
        return f'Prediction: {prediction[0]}'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
