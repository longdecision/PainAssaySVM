# -*- coding: utf-8 -*-
"""
svm20190122_generic_directory.py

5/7/2019 modified from svm20190122.py to use default directory and trim unused code

use svm directly on scaled data, not PCA

determine the transformation scalers and pca decomposition based on all data
specify which data to use for training the svm
TO define which genotype/gender to use for training, set file4train
TO specify which stimuli to use, set train_sets
TO define which features to use, set feature_names
TO specify how many pca components to use, set nPCA
TO specify the regularization term, set C_SVC 
@author: Long
"""
import numpy as np
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import style
style.use("seaborn-white")
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import os


def pipeline(df_settings):
    # standard pipeline for scaling features, computing PCA and training SVM
    # Common settings
    data_path = '.'    
    data_file = ['CD1 male.csv', 'CD1 female.csv', 'C57 male.csv', 'C57 female.csv', 'CD1 male VFH.csv', 'TrpV1-ChR2.csv', 'MrgD-ChR2.csv']
    out_path = 'PCA20190122'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    feature_names = ['Paw Velocity (mm/s)', 'Paw Height (mm)', 'Pain Score'] # features to be used for PCA computation
    train_sets = {'touch':['cotton swab'], 'pain':['heavy pinprick']} # determines which stimulus is included for touch/pain category for training
    C_SVC = 1
    nPCA = 1
    stimulus4pca_options = {'cs_hp': ['cotton swab', 'heavy pinprick'], 'all': None}
    
    flag_stimulus4pca = df_settings['flag_stimulus4pca']

    txt = df_settings['file4pca'].replace(']', '')
    txt = txt.replace('[', '')
    file4pca = [data_file[int(x)] for x in txt.split(',')]

    txt = df_settings['file4training'].replace(']', '')
    txt = txt.replace('[', '')
    file4train = [data_file[int(x)] for x in txt.split(',')]
 
    
    feature_SVM = ['pca1'] # use pca1 as the input for SVM
    
    stimulus4pca = stimulus4pca_options[flag_stimulus4pca]
    
    
    
    # construct output filename
    fout = 'PCA_' + flag_stimulus4pca + '_'
    for f in file4pca:
        fout += f[:-4].replace(' ', '') + ' '
    
    fout += '_Fit_'
    for f in file4train:
        fout += f[:-4] + '_'
    for stim in train_sets['touch'] + train_sets['pain']:
        s_words = stim.split()
        for s in s_words:
            fout += s[0]
        fout += '_'
    fout += 'c'+str(C_SVC) + 'pca' + str(nPCA)
    

    # estimate PCs using only data files specified in file4pca
    df4pca = collectData(data_path, file4pca, stimulus4pca)
    df4pca = df4pca.apply(lowercase, axis = 1)
    _, scalersTrain, pcaTrain, _ = scale_pca(df4pca, feature_names, scalers = None, pca = None, nPCA = nPCA)

    # collect all data into a big dataframe and calculate PC scores
    df = collectData(data_path, data_file)
    df = df.apply(lowercase, axis = 1)
    valPCA, scalersTrain, pcaTrain, valScaled = scale_pca(df, feature_names, scalers = scalersTrain, pca = pcaTrain, nPCA = nPCA)
    for i in range(nPCA):
        df['pca'+str(i+1)] = valPCA[:,i]
    for i, feature in enumerate(feature_names):
        df[feature + '_scaled'] = valScaled[:,i]
    
    # construct training data
    data2train = pd.DataFrame()
    type4train = [] # keep information about which types of data were included for the training dataset
    
    for f in file4train:
        for icat, catname in enumerate(['touch', 'pain']):
            for stim in train_sets[catname]:
                ind_this = (df.file == f)&(df.Stimulus == stim)
                data2include = pd.DataFrame(df.loc[ind_this])
                data2include['category'] = np.ones((len(data2include),1)) * icat
                data2train = data2train.append(data2include, ignore_index = True)
                type4train += [f +' '+ stim]
                df.loc[ind_this, 'used for training'] = 1
    ndata2train = len(data2train)


    print(fout)
    print('PCA files:' , file4pca, len(df4pca))
    print('SVM files:' , file4train, len(data2train))
    print(scalersTrain.mean_, '\n', scalersTrain.scale_, '\n', pcaTrain.components_[0])
    print('num of training trials:', ndata2train, '\n')


    # initialize and train a classifier
    colnames = list(data2train)
    colnames.remove('category')
    model = svm.SVC(C = C_SVC, kernel = 'rbf', probability = True)
    model.fit(data2train[feature_SVM], data2train.category)

    # predict from data
    prob = model.predict_proba(df[feature_SVM])


    df['p_pain']  = prob[:,1]

    # plot output
    n_data = len(data_file)
    fig, ax = plt.subplots(1,n_data)
    fig.set_size_inches(n_data*2.4,3)    
    for i, f in enumerate(data_file):
        plot_data(df.loc[df.file == f], ax[i], f, type4train)
    
    fig.autofmt_xdate()
    fig.tight_layout()
    
    df.to_csv(os.path.join(out_path, fout + '.csv'))
    fig.savefig( os.path.join(out_path, fout + '.pdf'), format = 'pdf')

    df_scaler = pd.DataFrame(np.array([scalersTrain.mean_, scalersTrain.scale_, pcaTrain.components_[0]]), columns = feature_names, index = ['mean', 'scaler', 'pca'])
    df_scaler.to_csv(os.path.join(out_path, fout + 'scaler.csv'))
    return(fout)

def main():
    

    # custom settings defined in setting_file
    settings_path = r'.'    
    setting_file = os.path.join(settings_path, 'plots_20190128.csv')
    df_settings = pd.read_csv(setting_file)
    
    fout = df_settings.apply(pipeline, axis = 1)
    df_settings['f_out'] = fout
    df_settings.to_csv(os.path.join(settings_path, 'log.csv'))           


def set_font_label(ax, ylabel, xlabel, title):    
    font0 = FontProperties()
    font = font0.copy()
    font.set_family('arial')
    font.set_size(8)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
    
    font.set_size(9)
    ax.set_ylabel(ylabel, fontproperties=font)
    ax.set_xlabel(xlabel, fontproperties=font, labelpad = 2)

    font.set_size(title[1])
    t = ax.set_title("\n".join(wrap(title[0], width = 26)), fontproperties=font)
    t.set_y(1.05)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params('both', direction = 'in', length=5, width=1, which='major')
    ax.locator_params(axis = 'X', nbins = 2)

def collectData(data_path, data_files, stimulus = None):
    df = pd.DataFrame()
    for f in data_files:
        df_this = pd.read_csv(os.path.join(data_path, f))
        df_this = df_this.dropna(axis=0, how = "all")
        df_this['file'] = [f]*len(df_this)
        df_this['Stimulus'] = df_this['Stimulus'].map(lambda x: x if type(x)!=str else x.lower())
        df = df.append(df_this)
    
    if stimulus is not None:
        df = df.loc[df.Stimulus.isin(stimulus)]
        
        
    return(df)

def lowercase(df):
    if type(df.Stimulus) == str:
        df.Stimulus = df.Stimulus.lower()
    return(df)

def scale_pca(df, features, scalers = None, pca = None, nPCA = 1):
    # scaling
    if scalers is None:
        scalers = preprocessing.StandardScaler()
        scalers.fit(df[features])
    X_scaled = scalers.transform(df[features])
    
    if nPCA >0:
        # pca decomposition
        if pca is None:
            pca = PCA(n_components= nPCA)
            pca.fit(X_scaled)
        X_PCA = pca.transform(X_scaled)

    return(X_PCA, scalers, pca, X_scaled)
    
def plot_data(df, ax, f, type4train):
    stim = df.Stimulus.unique()
    for i, stim_type in enumerate(stim):
        prob = df.loc[df.Stimulus == stim_type]['p_pain']
        x2plot = i + np.linspace(-0.2, 0.2, len(prob)) 
        if f + ' ' + str(stim_type) in type4train:
            ax.plot(x2plot, prob, marker = 'o', linestyle='None', markerfacecolor = 'None', markeredgecolor = 'k', clip_on = False)
        else:
            ax.plot(x2plot, prob, marker = 'o', linestyle='None', markerfacecolor = 'None', markeredgecolor = 'k', clip_on = False)
        ax.plot([i-0.3, i+0.3], [prob.mean(), prob.mean()], 'r-', marker='None', linestyle = '-', linewidth = 3 )
    ax.plot([-1, len(stim)], [0.5, 0.5], 'k--')
    ax.set_xlim([-1, len(stim)])
    ax.set_ylim([0,1])
    ax.set_xticks(np.arange(0,len(stim)).astype(int))
    ax.set_xticklabels(stim)
    set_font_label(ax, 'P_pain', '', [f[:-4], 9])
    
if __name__ == '__main__':
    main()