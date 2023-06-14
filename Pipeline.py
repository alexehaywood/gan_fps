from os import chdir
chdir('C:/Users/samue/Documents/GitHub/GAN_Scripts')

from GAN_Tools import get_default_device, to_device, GanComponent, Results, Environment
from GAN_Tools import classification_metrics

import pandas as pd
import numpy as np
from os import mkdir
from os.path import isdir, exists
from torch import optim
import random
from torch import manual_seed
from plotnine import ggplot, aes, geom_line, geom_vline, labels, scales
import joblib
import yaml

with open("./Data/microarray_config.yml", "r") as file:
    config = yaml.safe_load(file)
    
config_pre = config["pretraining"]

PATH = config["PATH"]

##Check dir structure, and make sub-directories if not present
if not isdir(PATH+'/Results'):
    mkdir(PATH+'/Results')
if not isdir(PATH+'/Results/External_Results'):
    mkdir(PATH+'/Results/External_Results')
if not isdir(PATH+'/Results/Main_Results'):
    mkdir(PATH+'/Results/Main_Results')
if not isdir(PATH+'/Results/External_Results/Saved_Pars'):
    mkdir(PATH+'/Results/External_Results/Saved_Pars')
if not isdir(PATH+'/Results/Main_Results/Saved_Pars'):
    mkdir(PATH+'/Results/Main_Results/Saved_Pars')
if not isdir(PATH+'/Results/Main_Results/overrep'):
    mkdir(PATH+'/Results/Main_Results/overrep')
if not isdir(PATH+'/Results/Main_Results/overrep/Saved_Pars'):
    mkdir(PATH+'/Results/Main_Results/overrep/Saved_Pars')
if not isdir(PATH+'/Data'):
    mkdir(PATH+'/Data')
    print('WARNING: ENSURE DATA IS COPIED INTO THIS DIRECTORY')


Env_pre = Environment()
Env_pre.device = get_default_device()

experiment_data = config["experiment_data"]

Env_pre.PATH = PATH

Env_pre.tune_params = dict(
    max_epochs_1=config_pre["max_epochs_1"], 
    max_loss_1=config_pre["max_loss_1"], 
    n_epochs_2=config_pre["n_epochs_2"], 
    dropout_prob=config_pre["dropout_prob"], 
    seed=config_pre["random_seed"], 
    lr_1 = config_pre["lr_1"],
    lr_2 = config_pre["lr_2"],              
    rate_save = config_pre["rate_save"], 
    diff_epochs = config_pre["diff_epochs"], 
    max_loss_2 = config_pre["max_loss_2"], 
    min_loss_2 = config_pre["min_loss_2"],
    alpha_instab = config_pre["alpha_instab"],
)
    
np.random.seed(config["random_seed"])
manual_seed(config["random_seed"])
random.seed(config["random_seed"])

#%%
###TRAINING
#pars
Env_pre.func_optim = getattr(optim, config_pre["func_optim"])
Env_pre.n_features = config_pre["n_features"]
Env_pre.alpha = config_pre["alpha"]
Env_pre.beta = config_pre["beta"]
Env_pre.batch_size = config_pre["batch_size"]
Env_pre.iter_critic = config_pre["iter_critic"]

dropout_prob = config_pre["dropout_prob"]

#%%
##remove IDs if real data
def prepare_dat(path_dir, file_name, n_cols_rem):
    if not exists(path_dir + '/' + file_name[:-4] + '_inp.csv'):
        dat_temp = pd.read_csv(path_dir + '/' + file_name)
        dat_temp = dat_temp.iloc[:, n_cols_rem:]
        dat_temp.to_csv(path_dir + '/' + file_name[:-4] + '_inp.csv', index = False)

#remove ID and label
prepare_dat(PATH + '/Data', 'dat_ext.csv', config_pre["data_prep"][experiment_data])    
     
#%%
##Train pre_GAN on external data

#For sim
#Env_pre.path_dat = PATH + '/sim_dat_ext.csv'
#For real
Env_pre.path_dat = PATH + '/Data/dat_ext_inp.csv'

Env_pre.path_results = PATH + '/Results/External_Results'
Env_pre.path_pars = Env_pre.path_results + '/Saved_Pars'

gen_struct = config_pre["gen_structure"]
critic_struct = config_pre["critic_structure"]

Env_pre.n_training_samples = sum(1 for line in open(Env_pre.path_dat)) - 1 #number of records in file (excludes header)

Env_pre.critic = GanComponent(Env_pre.n_features, 
                              Env_pre.init_weights, 
                              critic_struct[0], 
                              dropout_prob,
                              "critic")
Env_pre.generator = GanComponent(Env_pre.n_features, 
                                 Env_pre.init_weights, 
                                 gen_struct[0], 
                                 dropout_prob,
                                 "generator")
Env_pre.results_record = Results(path = Env_pre.path_results, saved_results = False)

to_device(Env_pre.critic.train(), Env_pre.device)
to_device(Env_pre.generator.train(), Env_pre.device)


##transformer generated based on batch of size 200 or max data if < 200 (95 samples in metabolomics)
if experiment_data == 'metabolomics':
    Env_pre.transformer = Env_pre.get_scaler(95, '/ext_transformer.pkl')
else:
    Env_pre.transformer = Env_pre.get_scaler(200, '/ext_transformer.pkl')



#%%
##First training loop
#if experiment_data == 'microarray':
#    Env_pre.lr = 0.0005
#    n_epochs = 1000
#if experiment_data == 'simulation':
#    Env_pre.lr = 0.001
#    n_epochs = 1000
#if experiment_data == 'metabolomics':
#    Env_pre.lr = 0.001
#    n_epochs = 1000

max_loss = 15
min_loss = -10
    
##First training loop
telapsed_summary_1 = Env_pre.tuneAndTrain(max_loss, min_loss, dropout_prob, preTrained=False)

Env_pre.generator.grow(gen_struct[1])
Env_pre.critic.grow(critic_struct[1])

to_device(Env_pre.generator, Env_pre.device)
to_device(Env_pre.critic, Env_pre.device)

##Second training loop
telapsed_summary_2 = Env_pre.tuneAndTrain(max_loss, min_loss, dropout_prob, preTrained=False)

Env_pre.generator.grow(gen_struct[2])
Env_pre.critic.grow(critic_struct[2])

to_device(Env_pre.generator, Env_pre.device)
to_device(Env_pre.critic, Env_pre.device)

##Third training loop
#input
Env_pre.iter_critic = 2
dropout_prob = 0.7

Env_pre.generator.change_dropout(dropout_prob)
Env_pre.critic.change_dropout(dropout_prob)

telapsed_summary_3 = Env_pre.tuneAndTrain(max_loss, min_loss, dropout_prob, preTrained=False)

##Save time stats
dat_telapsed = pd.concat([telapsed_summary_1, telapsed_summary_2, telapsed_summary_3], axis=1)
dat_telapsed.to_csv(Env_pre.PATH + '/Results/telapsed.csv')

###############################################################################################

config_pre = config["pretraining"]

PATH = config["PATH"]

##Check dir structure, and make sub-directories if not present
if not isdir(PATH+'/Results'):
    mkdir(PATH+'/Results')
if not isdir(PATH+'/Results/External_Results'):
    mkdir(PATH+'/Results/External_Results')
if not isdir(PATH+'/Results/Main_Results'):
    mkdir(PATH+'/Results/Main_Results')
if not isdir(PATH+'/Results/External_Results/Saved_Pars'):
    mkdir(PATH+'/Results/External_Results/Saved_Pars')
if not isdir(PATH+'/Results/Main_Results/Saved_Pars'):
    mkdir(PATH+'/Results/Main_Results/Saved_Pars')
if not isdir(PATH+'/Results/Main_Results/overrep'):
    mkdir(PATH+'/Results/Main_Results/overrep')
if not isdir(PATH+'/Results/Main_Results/overrep/Saved_Pars'):
    mkdir(PATH+'/Results/Main_Results/overrep/Saved_Pars')
if not isdir(PATH+'/Data'):
    mkdir(PATH+'/Data')
    print('WARNING: ENSURE DATA IS COPIED INTO THIS DIRECTORY')


Env_pre = Environment()
Env_pre.device = get_default_device()

experiment_data = config["experiment_data"]

Env_pre.PATH = PATH

Env_pre.tune_params = dict(
    max_epochs_1=config_pre["max_epochs_1"], 
    max_loss_1=config_pre["max_loss_1"], 
    n_epochs_2=config_pre["n_epochs_2"], 
    dropout_prob=config_pre["dropout_prob"], 
    seed=config_pre["random_seed"], 
    lr_1 = config_pre["lr_1"],
    lr_2 = config_pre["lr_2"],              
    rate_save = config_pre["rate_save"], 
    diff_epochs = config_pre["diff_epochs"], 
    max_loss_2 = config_pre["max_loss_2"], 
    min_loss_2 = config_pre["min_loss_2"],
    alpha_instab = config_pre["alpha_instab"],
)
    
np.random.seed(config["random_seed"])
manual_seed(config["random_seed"])
random.seed(config["random_seed"])

#%%
###TRAINING
#pars
Env_pre.func_optim = getattr(optim, config_pre["func_optim"])
Env_pre.n_features = config_pre["n_features"]
Env_pre.alpha = config_pre["alpha"]
Env_pre.beta = config_pre["beta"]
Env_pre.batch_size = config_pre["batch_size"]
Env_pre.iter_critic = config_pre["iter_critic"]

dropout_prob = config_pre["dropout_prob"]

#%%
##remove IDs if real data
def prepare_dat(path_dir, file_name, n_cols_rem):
    if not exists(path_dir + '/' + file_name[:-4] + '_inp.csv'):
        dat_temp = pd.read_csv(path_dir + '/' + file_name)
        dat_temp = dat_temp.iloc[:, n_cols_rem:]
        dat_temp.to_csv(path_dir + '/' + file_name[:-4] + '_inp.csv', index = False)

#remove ID and label
prepare_dat(PATH + '/Data', 'dat_ext.csv', config_pre["data_prep"][experiment_data])    
     
#%%
##Train pre_GAN on external data

#For sim
#Env_pre.path_dat = PATH + '/sim_dat_ext.csv'
#For real
Env_pre.path_dat = PATH + '/Data/dat_ext_inp.csv'

Env_pre.path_results = PATH + '/Results/External_Results'
Env_pre.path_pars = Env_pre.path_results + '/Saved_Pars'

gen_struct = config_pre["gen_structure"]
critic_struct = config_pre["critic_structure"]

Env_pre.n_training_samples = sum(1 for line in open(Env_pre.path_dat)) - 1 #number of records in file (excludes header)

Env_pre.critic = GanComponent(Env_pre.n_features, 
                              Env_pre.init_weights, 
                              critic_struct[0], 
                              dropout_prob,
                              "critic")
Env_pre.generator = GanComponent(Env_pre.n_features, 
                                 Env_pre.init_weights, 
                                 gen_struct[0], 
                                 dropout_prob,
                                 "generator")
Env_pre.results_record = Results(path = Env_pre.path_results, saved_results = False)

to_device(Env_pre.critic.train(), Env_pre.device)
to_device(Env_pre.generator.train(), Env_pre.device)


##transformer generated based on batch of size 200 or max data if < 200 (95 samples in metabolomics)
if experiment_data == 'metabolomics':
    Env_pre.transformer = Env_pre.get_scaler(95, '/ext_transformer.pkl')
else:
    Env_pre.transformer = Env_pre.get_scaler(200, '/ext_transformer.pkl')



#%%
##First training loop
#if experiment_data == 'microarray':
#    Env_pre.lr = 0.0005
#    n_epochs = 1000
#if experiment_data == 'simulation':
#    Env_pre.lr = 0.001
#    n_epochs = 1000
#if experiment_data == 'metabolomics':
#    Env_pre.lr = 0.001
#    n_epochs = 1000

max_loss = 15
min_loss = -10
    
##First training loop
telapsed_summary_1 = Env_pre.tuneAndTrain(max_loss, min_loss, dropout_prob, preTrained=False)

Env_pre.generator.grow(gen_struct[1])
Env_pre.critic.grow(critic_struct[1])

to_device(Env_pre.generator, Env_pre.device)
to_device(Env_pre.critic, Env_pre.device)

##Second training loop
telapsed_summary_2 = Env_pre.tuneAndTrain(max_loss, min_loss, dropout_prob, preTrained=False)

Env_pre.generator.grow(gen_struct[2])
Env_pre.critic.grow(critic_struct[2])

to_device(Env_pre.generator, Env_pre.device)
to_device(Env_pre.critic, Env_pre.device)

##Third training loop
#input
Env_pre.iter_critic = 2
dropout_prob = 0.7

Env_pre.generator.change_dropout(dropout_prob)
Env_pre.critic.change_dropout(dropout_prob)

telapsed_summary_3 = Env_pre.tuneAndTrain(max_loss, min_loss, dropout_prob, preTrained=False)

##Save time stats
dat_telapsed = pd.concat([telapsed_summary_1, telapsed_summary_2, telapsed_summary_3], axis=1)
dat_telapsed.to_csv(Env_pre.PATH + '/Results/telapsed.csv')