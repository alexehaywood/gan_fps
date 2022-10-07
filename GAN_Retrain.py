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

    max_loss = 15
    min_loss = -10
    if path_ID[-1] != 'e':
        Env_under.load_pre(dropout_prob, 118)
        lr_1, n_epochs_1, lr_2, results_tuning_1, results_tuning_2 = Env_under.auto_tune(5000, 3, 5000, dropout_prob, 118, diff_epochs = 500,
                                                                                         max_loss_2=max_loss, min_loss_2=min_loss)
    else:
        Env_under.load_pre(dropout_prob, 118, preTrained = False)
        lr_1, n_epochs_1, lr_2, results_tuning_1, results_tuning_2 = Env_under.auto_tune(5000, 3, 5000, dropout_prob, 118, diff_epochs = 500,
                                                                                         max_loss_2=max_loss, min_loss_2=min_loss, 
                                                                                         preTrained = False)


    results_tuning_1 = pd.DataFrame.from_dict(results_tuning_1)
    results_tuning_2 = pd.DataFrame.from_dict(results_tuning_2)
    results_tuning_1.to_csv(Env_under.path_results + '/results_tuning_1_under')
    results_tuning_2.to_csv(Env_under.path_results + '/results_tuning_2_under')


    ##Train with real data
    print('Training')
    Env_under.lr = lr_1
    telapsed_1 = Env_under.train(n_epochs_1)

    Env_under.lr = lr_2
    telapsed_2 = Env_under.train(10000-n_epochs_1)

    #save times
    telapsed_summary = pd.DataFrame([ telapsed_1, telapsed_2 ])
    telapsed_summary.to_csv(Env_under.path_results + "/time_training.csv")


    ##Select final generator pars
    ##check for null values
    losses = Env_under.results_record.record['loss_critic']
    ind_check = np.array([np.nan, np.nan])

    if losses.gt(max_loss).sum() > 0: #check for null values
        ind_check[0] = losses.gt(max_loss).argmax()
    if losses.lt(min_loss).sum() > 0:
        ind_check[1] = losses.lt(min_loss).argmax()

    if np.isnan(ind_check).sum() == 2: #no values gt or lt limits
        ind_null = len(losses)
    else:
        ind_null = int(ind_check[np.nanargmin(ind_check)]) + 1

    losses = losses[:ind_null]
    print('Null value at index {}, epoch {}'.format(ind_null, ind_null * 20))

    nEpoch_pars, graph_loss = Env_under.findApEl(offset = ind_null)

    plot = (
        ggplot(Env_under.results_record.record.iloc[:ind_null-1])
        + geom_line(color = 'red', mapping = aes(y = 'loss_critic', x = 'epoch'))
        + geom_line(color = 'blue', mapping = aes(y = 'loss_gen', x = 'epoch'))
        + geom_vline(xintercept = nEpoch_pars, size = 0.2)
        + labels.xlab('Epoch')
        + labels.ylab('Loss')
        + scales.ylim(-10, 15)
    )

    graph_loss.save(Env_under.path_results+'/plot_loss_bestFit' + '_retrain' + '.png')
    plot.save(Env_under.path_results+'/plot_loss' + '_retrain' + '.png')
    Env_under.select_pars(str(nEpoch_pars))
    Env_under.generator.save_pars(Env_under.path_pars, 'complete_retrain')

    Env_under.results_record.save_results(Env_under.PATH + path_ID + '/Results')
  




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


    max_loss = 15
    min_loss = -10
    if path_ID[-1] != 'e':
        Env_over.load_pre(dropout_prob, 119)
        lr_1, n_epochs_1, lr_2, results_tuning_1, results_tuning_2 = Env_over.auto_tune(5000, 3, 5000, dropout_prob, 118, diff_epochs = 500,
                                                                                         max_loss_2=max_loss, min_loss_2=min_loss)
    else:
        Env_over.load_pre(dropout_prob, 119, preTrained = False)
        lr_1, n_epochs_1, lr_2, results_tuning_1, results_tuning_2 = Env_over.auto_tune(5000, 3, 5000, dropout_prob, 118, diff_epochs = 500,
                                                                                         max_loss_2=max_loss, min_loss_2=min_loss, 
                                                                                         preTrained = False)

    results_tuning_1 = pd.DataFrame.from_dict(results_tuning_1)
    results_tuning_2 = pd.DataFrame.from_dict(results_tuning_2)
    results_tuning_1.to_csv(Env_over.path_results + '/results_tuning_1_under')
    results_tuning_2.to_csv(Env_over.path_results + '/results_tuning_2_under')


    ##Train with real data
    print('Training')
    Env_over.lr = lr_1
    telapsed_1 = Env_over.train(n_epochs_1)

    Env_over.lr = lr_2
    telapsed_2 = Env_over.train(10000-n_epochs_1)

    #save times
    telapsed_summary = pd.DataFrame([ telapsed_1, telapsed_2 ])
    telapsed_summary.to_csv(Env_over.path_results + "/time_training.csv")


    ##Select final generator pars
    losses = Env_over.results_record.record['loss_critic']
    ind_check = np.array([np.nan, np.nan])

    if losses.gt(max_loss).sum() > 0: #check for null values
        ind_check[0] = losses.gt(max_loss).argmax()
    if losses.lt(min_loss).sum() > 0:
        ind_check[1] = losses.lt(min_loss).argmax()

    if np.isnan(ind_check).sum() == 2: #no values gt or lt limits
        ind_null = len(losses)
    else:
        ind_null = int(ind_check[np.nanargmin(ind_check)]) + 1

    losses = losses[:ind_null]
    print('Null value at index {}, epoch {}'.format(ind_null, ind_null * 20))

    nEpoch_pars, graph_loss = Env_over.findApEl(offset = ind_null)

    plot = (
        ggplot(Env_over.results_record.record.iloc[:ind_null-1])
        + geom_line(color = 'red', mapping = aes(y = 'loss_critic', x = 'epoch'))
        + geom_line(color = 'blue', mapping = aes(y = 'loss_gen', x = 'epoch'))
        + geom_vline(xintercept = nEpoch_pars, size = 0.2)
        + labels.xlab('Epoch')
        + labels.ylab('Loss')
        + scales.ylim(-10, 15)
    )


    graph_loss.save(Env_over.path_results+'/plot_loss_bestFit' + '_retrain' + '.png')
    plot.save(Env_over.path_results+'/plot_loss' + '_retrain' + '.png')
    Env_over.select_pars(str(nEpoch_pars))
    Env_over.generator.save_pars(Env_over.path_pars, 'complete_retrain')

    Env_over.results_record.save_results(Env_over.PATH + path_ID + '/Results')

   



    #Load underrep GAN for validation
    pre_generator = Generator(Env_under.n_features, Env_under.init_weights, 50, dropout_prob)
    pre_generator.grow(100)
    pre_generator.grow(200)

    pre_generator.load_pars(Env_under.path_pars, 'complete_retrain', Env_under.device)
    pre_generator.change_dropout(dropout_prob)

    pre_generator.eval()
    to_device(pre_generator, Env_under.device)


    #Load overrep GAN for validation
    pre_generator2 = Generator(Env_over.n_features, Env_under.init_weights, 50, dropout_prob)
    pre_generator2.grow(100)
    pre_generator2.grow(200)

    pre_generator2.load_pars(Env_over.path_pars, 'complete_retrain', Env_over.device)
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
