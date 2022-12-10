import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


def plot_correlation_matrix(df):
    # creating a list of only numerical values
    numerical = [u'Age', u'DailyRate', u'DistanceFromHome', 
                u'Education', u'EnvironmentSatisfaction',
                u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction',
                u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',
                u'PercentSalaryHike', u'PerformanceRating',
                u'StockOptionLevel', u'TotalWorkingYears',
                u'TrainingTimesLastYear', u'WorkLifeBalance', u'YearsAtCompany',
                u'YearsInCurrentRole', u'YearsSinceLastPromotion',u'YearsWithCurrManager']
    data = [
        go.Heatmap(
            z= df[numerical].astype(float).corr().values, # Generating the Pearson correlation
            x=df[numerical].columns.values,
            y=df[numerical].columns.values,
            colorscale='Viridis',
            reversescale = False,
    #         text = True ,
            opacity = 1.0
            
        )
    ]


    layout = go.Layout(
        title='Pearson Correlation of numerical features',
        xaxis = dict(ticks='', nticks=36),
        yaxis = dict(ticks='' ),
        width = 900, height = 700,
        
    )


    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='labelled-heatmap')

def display_metrics(score):
    print("Accuracy : {}".format(score["test_accuracy"].mean()))
    print("Recall : {}".format(score["test_recall"].mean()))
    print("F1 : {}".format(score["test_f1"].mean()))