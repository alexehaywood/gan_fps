from os import chdir
chdir('...')

from GAN_Tools import get_default_device, to_device, Generator, Results, Environment

import pandas as pd
import numpy as np
from os import mkdir
from os.path import isdir
from torch import optim
import random
from torch import manual_seed
import joblib
from plotnine import ggplot, aes, geom_line, geom_vline, labels, scales

#input##########################
PATH = '...'

###Which Data
#experiment_data = 'microarray'
#experiment_data = 'simulation'
experiment_data = 'metabolomics'

#%%
################################
##Check dir structure

if experiment_data == 'metabolomics':
    dirs = [
        '/1a', '/1b', '/1c'
        ]
else:
    dirs = [
        '/1a', '/1b', '/1c',
        '/2a', '/2b', '/2c',
        '/3a', '/3b', '/3c',
        '/4a', '/4b', '/4c',
        '/5a', '/5b', '/5c',
        '/6a', '/6b', '/6c',
        '/7a', '/7b', '/7c',
        '/8a', '/8b', '/8c',
        '/9a', '/9b', '/9c',
        '/1-Compare', '/2-Compare', '/3-Compare',
        '/4-Compare', '/5-Compare', '/6-Compare', 
        '/7-Compare', '/8-Compare', '/9-Compare'
        ]

for path_ID in dirs:
    ##Check dir structure, and make sub-directories if not present
    if not isdir(PATH+ path_ID+'/Results'):
        mkdir(PATH+path_ID+'/Results')
    if not isdir(PATH+path_ID+'/Results/External_Results'):
        mkdir(PATH+path_ID+'/Results/External_Results')
    if not isdir(PATH+path_ID+'/Results/Main_Results'):
        mkdir(PATH+path_ID+'/Results/Main_Results')
    if not isdir(PATH+path_ID+'/Results/Main_Results/underrep'):
        mkdir(PATH+path_ID+'/Results/Main_Results/underrep')
    if not isdir(PATH+path_ID+'/Results/External_Results/Saved_Pars'):
        mkdir(PATH+path_ID+'/Results/External_Results/Saved_Pars')
    if not isdir(PATH+path_ID+'/Results/Main_Results/underrep/Saved_Pars'):
        mkdir(PATH+path_ID+'/Results/Main_Results/underrep/Saved_Pars')
    if not isdir(PATH+path_ID+'/Results/Main_Results/overrep'):
        mkdir(PATH+path_ID+'/Results/Main_Results/overrep')
    if not isdir(PATH+path_ID+'/Results/Main_Results/overrep/Saved_Pars'):
        mkdir(PATH+path_ID+'/Results/Main_Results/overrep/Saved_Pars')
    if not isdir(PATH+path_ID+'/Data'):
        mkdir(PATH+path_ID+'/Data')
        print('WARNING: ENSURE DATA IS COPIED INTO THIS DIRECTORY')

#%%

max_loss = 15
min_loss = -10

for path_ID in dirs:

    Env_under = Environment()
    Env_under.device = get_default_device()
    Env_under.PATH = PATH


    Env_over = Environment()
    Env_over.device = get_default_device()
    Env_over.PATH = PATH

    ##Input ################################
    Env_under.n_features, Env_over.n_features = 218, 218
    Env_under.func_optim, Env_over.func_optim = optim.SGD, optim.SGD

    Env_under.batch_size, Env_over.batch_size = 10, 20
    Env_under.iter_critic, Env_over.iter_critic = 2, 2
    Env_under.beta, Env_over.beta = 13, 13
    
    Env_under.structure, Env_over.structure = [50, 100, 200], [50, 100, 200]

    dropout_prob = 0.7

    if path_ID[-1] == 'a':
        Env_under.alpha, Env_over.alpha = 0, 0
    elif path_ID[-1] == 'b':
        Env_under.alpha, Env_over.alpha = 1, 0
    elif path_ID[-1] == 'c':
        Env_under.alpha, Env_over.alpha = 1, 1
    elif path_ID[-1] == 'e':
        Env_under.alpha, Env_over.alpha = 0, 0
    else:
        print('error')
        break
        
    
    ########################################



    ##remove IDs if microarray data
    def prepare_dat(path_dir, file_name, n_cols_rem):
        dat_temp = pd.read_csv(path_dir + '/' + file_name)
        dat_temp = dat_temp.iloc[:, n_cols_rem:]
        dat_temp.to_csv(path_dir + '/' + file_name[:-4] + '_inp.csv', index = False)

    #remove ID and label
    if experiment_data == 'microarray':
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_class1.csv', 2)
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_combo.csv', 2)
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_class2.csv', 2)
    if experiment_data == 'simulation':
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_class1.csv', 0)
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_combo.csv', 1)
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_class2.csv', 0)
    if experiment_data == 'metabolomics':
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_class1.csv', 2)
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_combo.csv', 2)
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_class2.csv', 2)





    ################################################################################
    ##RE-training loop

    ##Prepare data
    Env_under.path_dat = PATH + path_ID + '/Data/dat_real_combo_inp.csv'
    Env_under.path_results = PATH + path_ID + '/Results/Main_Results/underrep'
    Env_under.path_pars = Env_under.path_results + '/Saved_Pars'


    Env_under.results_record = Results(path = Env_under.path_results, saved_results = False)

    ##transformer based on all data (Combo), due to limited available
    Env_under.transformer = Env_under.get_scaler( (sum(1 for line in open(Env_under.path_dat)) - 1), '/real_transformer.pkl')


    if experiment_data == 'microarray' or experiment_data == 'metabolomics':
        Env_under.path_dat = PATH + path_ID + '/Data/dat_real_class1_inp.csv'
    if experiment_data == 'simulation':
        Env_under.path_dat = PATH + path_ID + '/Data/dat_real_class1.csv'

    Env_under.n_training_samples = sum(1 for line in open(Env_under.path_dat)) - 1 #number of records in file (excludes header)

    if path_ID[-1] != 'e':
        telapsed_summary = Env_under.tuneAndTrain(max_loss, min_loss)
    else:
        telapsed_summary = Env_under.tuneAndTrain(max_loss, min_loss, preTrained=False)
    #save times
    telapsed_summary.to_csv(Env_under.path_results + "/time_training.csv")

    ################################################################################
    ##RE-training loop

    ##Prepare data
    if experiment_data == 'simulation':
        Env_over.path_dat = Env_over.PATH + path_ID + '/Data/dat_real_class2.csv'
    if experiment_data == 'microarray' or experiment_data == 'metabolomics':
        Env_over.path_dat = Env_over.PATH + path_ID + '/Data/dat_real_class2_inp.csv'

    Env_over.path_results = Env_over.PATH + path_ID + '/Results/Main_Results/overrep'
    Env_over.path_pars = Env_over.path_results + '/Saved_Pars'

    n_training_samples = sum(1 for line in open(Env_over.path_dat)) - 1 #number of records in file (excludes header)

    Env_over.results_record = Results(path = Env_over.path_results, saved_results = False)

    #load combo transformer
    Env_over.transformer = joblib.load(Env_under.path_results + '/real_transformer.pkl')

    Env_over.n_training_samples = sum(1 for line in open(Env_under.path_dat)) - 1 #number of records in file (excludes header)

    if path_ID[-1] != 'e':
        telapsed_summary = Env_under.tuneAndTrain(max_loss, min_loss)
    else:
        telapsed_summary = Env_under.tuneAndTrain(max_loss, min_loss, preTrained=False)
    #save times
    telapsed_summary.to_csv(Env_under.path_results + "/time_training.csv")

    #############################################################################################################
    #Load underrep GAN for validation
    pre_generator = Generator(Env_under.n_features, Env_under.init_weights, 50, dropout_prob)
    pre_generator.grow(100)
    pre_generator.grow(200)

    pre_generator.load_pars(Env_under.path_pars, f"{pre_generator.n_pre_layers}Retraining", Env_under.device)
    pre_generator.change_dropout(dropout_prob)

    pre_generator.eval()
    to_device(pre_generator, Env_under.device)


    #Load overrep GAN for validation
    pre_generator2 = Generator(Env_over.n_features, Env_under.init_weights, 50, dropout_prob)
    pre_generator2.grow(100)
    pre_generator2.grow(200)

    pre_generator2.load_pars(Env_over.path_pars, f"{pre_generator2.n_pre_layers}Retraining", Env_over.device)
    pre_generator2.change_dropout(dropout_prob)

    pre_generator2.eval()
    to_device(pre_generator2, Env_over.device)

    ##Validation
    from GAN_Tools import classification_metrics

    if experiment_data == 'microarray':
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_combo.csv', 1) #remove ID, keep label
        prepare_dat(PATH + '/Data', 'dat_val.csv', 2) #remove obsolete col and ID
    if experiment_data == 'simulation':
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_combo.csv', 0)
        prepare_dat(PATH + '/Data', 'dat_val.csv', 0)
    if experiment_data == 'metabolomics':
        prepare_dat(PATH + path_ID + '/Data', 'dat_real_combo.csv', 1)
        prepare_dat(PATH + '/Data', 'dat_val.csv', 1)

    np.random.seed(120)
    manual_seed(120)
    random.seed(120)

    #input##########################
    if experiment_data == 'simulation':
        path_dat_train = PATH + path_ID + '/Data/dat_real_combo_inp.csv'
        path_dat_val = PATH + '/Data/dat_val_inp.csv'
    if experiment_data == 'microarray' or experiment_data == 'metabolomics':
        path_dat_train = PATH + path_ID + '/Data/dat_real_combo_inp.csv'
        path_dat_val = PATH + '/Data/dat_val_inp.csv'
    ################################

    path_transformer = PATH + path_ID + '/Results/Main_Results/underrep' + '/real_transformer.pkl'


    #the case label must be 1st alphabetically due to functionality of labelEncoder
    classification_metrics(path_dat_train, path_dat_val, pre_generator, path_transformer,
                           Env_under.n_features, Env_under.device,
                           PATH + path_ID,
                           pre_generator2)
