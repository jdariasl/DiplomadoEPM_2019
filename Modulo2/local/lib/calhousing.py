import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
import warnings

from bokeh.plotting import *
#from bokeh.charts import *
from bokeh.models import WMTSTileSource
from bokeh.tile_providers import *
import bokeh

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

output_notebook(resources=bokeh.resources.INLINE)
warnings.filterwarnings('ignore')


def latlng_to_meters(lat, lng):
    origin_shift = 2 * np.pi * 6378137 / 2.0
    mx = lng * origin_shift / 180.0
    my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    my = my * origin_shift / 180.0
    return mx, my


def gridsearch_best(X,y, estimator, parameters, n_iter=10, test_size=0.3):
    cv = ShuffleSplit(n_splits=n_iter, test_size=test_size, random_state=0)
    clf = GridSearchCV(estimator, parameters, cv=cv, scoring=rel_rmse)
    gs = clf.fit(X,y)
    k = pd.DataFrame(gs.cv_results_)
    best = k.iloc[np.argmax(k.rank_test_score)]
    return best, k


def plot_learning_curve(estimator, title, X, y, ylim=None, n_iter=10, test_size=.3,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    cv = ShuffleSplit(n_splits=n_iter, test_size=test_size, random_state=0)
    scoring = rel_rmse
    
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title(title +"  score: %.2f"%test_scores_mean[-1]+" len data=%d"%len(X))
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="test score")

    plt.legend(loc="best")

def plot_map(lat, lon, color=None, size=10):
    cmap = cm.rainbow
    wlat, wlong = latlng_to_meters(lat, lon)
    if color is not None:
        colors = MinMaxScaler(feature_range=(0,255)).fit_transform(color)
        colors = ["#%02x%02x%02x"%tuple([int(j*255) for j in cmap(int(i))[:3]]) for i in colors]

    openmap_url = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'
    otile_url = 'http://otile1.mqcdn.com/tiles/1.0.0/sat/{Z}/{X}/{Y}.jpg'

    TILES = WMTSTileSource(url=openmap_url)
    tools="pan,wheel_zoom,reset"
    p = figure(tools=tools, plot_width=700,plot_height=600)

    p.add_tile(TILES)
    p.circle(np.array(wlat), np.array(wlong), color=colors, size=size)

    p.axis.visible = False
    
    cb = figure(plot_width=130, plot_height=600,  tools=tools)
    yc = np.linspace(np.min(color),np.max(color),20)
    c = np.linspace(0,255,20).astype(int)
    dy = yc[1]-yc[0]    
    cb.rect(x=0.5, y=yc, color=["#%02x%02x%02x"%tuple([int(j*255) for j in cmap(int(i))[:3]]) for i in c], width=1, height = dy)
    cb.xaxis.visible = False
    pb = gridplot([[p, cb]])
    show(pb)
        
    
def rmse(estimator, X, y):
    preds = estimator.predict(X)
    return np.sqrt(np.mean((preds-y)**2))

def rel_rmse(estimator, X, y):
    preds = estimator.predict(X)
    return np.mean(np.abs(preds-y)/y)
