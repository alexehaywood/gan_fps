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

with open("./microarray_config.yml", "r") as file:
    config = yaml.safe_load(file)
    
###############################################################################
if config["train_pre"]:
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

    ##First training loop
    telapsed_summary_1 = Env_pre.tuneAndTrain(preTrained=False)

    Env_pre.generator.grow(gen_struct[1])
    Env_pre.critic.grow(critic_struct[1])

    to_device(Env_pre.generator, Env_pre.device)
    to_device(Env_pre.critic, Env_pre.device)

    ##Second training loop
    telapsed_summary_2 = Env_pre.tuneAndTrain(preTrained=False)

    Env_pre.generator.grow(gen_struct[2])
    Env_pre.critic.grow(critic_struct[2])

    to_device(Env_pre.generator, Env_pre.device)
    to_device(Env_pre.critic, Env_pre.device)

    ##Third training loop
    #input
    Env_pre.iter_critic = 2
    dropout_prob = 0.7

    Env_pre.tune_params["dropout_prob"] = dropout_prob

    Env_pre.generator.change_dropout(dropout_prob)
    Env_pre.critic.change_dropout(dropout_prob)

    telapsed_summary_3 = Env_pre.tuneAndTrain(preTrained=False)

    ##Save time stats
    dat_telapsed = pd.concat([telapsed_summary_1, telapsed_summary_2, telapsed_summary_3], axis=1)
    dat_telapsed.to_csv(Env_pre.PATH + '/Results/telapsed.csv')

##########################################################################

#input##########################
PATH = config["PATH"]

config_re = config["retraining"]
###Which Data
experiment_data = config["experiment_data"]

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
    #if not isdir(PATH+ path_ID):
    #    mkdir(PATH+path_ID)
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

func_optim_under = getattr(optim, config_re["func_optim"][0])
func_optim_over = getattr(optim, config_re["func_optim"][1])

for path_ID in dirs:

    Env_under, Env_over = Environment(), Environment()
    Env_under.device, Env_over.device = get_default_device(), get_default_device()
    Env_under.PATH, Env_over.PATH = PATH, PATH
    Env_under.path_prePars, Env_over.path_prePars = config_re["path_prePars"], config_re["path_prePars"]
    
    ##Input ################################
    Env_under.n_features, Env_over.n_features = config_re["n_features"][0], config_re["n_features"][0]
    Env_under.func_optim, Env_over.func_optim = func_optim_under, func_optim_over

    Env_under.batch_size, Env_over.batch_size = config_re["batch_size"][0], config_re["batch_size"][1]
    Env_under.iter_critic, Env_over.iter_critic = config_re["iter_critic"][0], config_re["iter_critic"][1]
    Env_under.beta, Env_over.beta = config_re["beta"][0], config_re["beta"][1]
    
    dropout_prob = config_re["dropout_prob"][0]

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
        
    Env_under.tune_params = dict(
        max_epochs_1=config_re["max_epochs_1"][0], 
        max_loss_1=config_re["max_loss_1"][0], 
        n_epochs_2=config_re["n_epochs_2"][0], 
        dropout_prob=config_re["dropout_prob"][0], 
        seed=config["random_seed"], 
        lr_1 = config_re["lr_1"][0],
        lr_2 = config_re["lr_2"][0],              
        rate_save = config_re["rate_save"][0], 
        diff_epochs = config_re["diff_epochs"][0], 
        max_loss_2 = config_re["max_loss_2"][0], 
        min_loss_2 = config_re["min_loss_2"][0],
        alpha_instab = config_re["alpha_instab"][0],
    )
    Env_over.tune_params = dict(
        max_epochs_1=config_re["max_epochs_1"][1], 
        max_loss_1=config_re["max_loss_1"][1], 
        n_epochs_2=config_re["n_epochs_2"][1], 
        dropout_prob=config_re["dropout_prob"][1], 
        seed=config["random_seed"], 
        lr_1 = config_re["lr_1"][1],
        lr_2 = config_re["lr_2"][1],              
        rate_save = config_re["rate_save"][1], 
        diff_epochs = config_re["diff_epochs"][1], 
        max_loss_2 = config_re["max_loss_2"][1], 
        min_loss_2 = config_re["min_loss_2"][1],
        alpha_instab = config_re["alpha_instab"][1],
    )
    ######################################################
    ##remove IDs and label
    def prepare_dat(path_dir, file_name, n_cols_rem):
        dat_temp = pd.read_csv(path_dir + '/' + file_name)
        dat_temp = dat_temp.iloc[:, n_cols_rem:]
        dat_temp.to_csv(path_dir + '/' + file_name[:-4] + '_inp.csv', index = False)

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
    ##UNDER RE-training loop
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
    
    Env_under.generator = GanComponent(Env_under.n_features, 
                              Env_under.init_weights, 
                              config_re["gen_structure"][0][0], 
                              dropout_prob,
                              "generator")
    Env_under.critic = GanComponent(Env_under.n_features, 
                              Env_under.init_weights, 
                              config_re["critic_structure"][0][0], 
                              dropout_prob,
                              "critic")
    Env_under.generator.grow(config_re["gen_structure"][0][1])
    Env_under.generator.grow(config_re["gen_structure"][0][2])
    
    Env_under.critic.grow(config_re["critic_structure"][0][1])
    Env_under.critic.grow(config_re["critic_structure"][0][2])
    
    if path_ID[-1] != 'e':
        telapsed_summary = Env_under.tuneAndTrain()
    else:
        telapsed_summary = Env_under.tuneAndTrain(preTrained=False)
    #save times
    telapsed_summary.to_csv(Env_under.path_results + "/time_training.csv")

    ################################################################################
    ##OVER RE-training loop
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
    
    Env_over.generator = GanComponent(Env_over.n_features, 
                              Env_over.init_weights, 
                              config_re["gen_structure"][1][0], 
                              dropout_prob,
                              "generator")
    Env_over.critic = GanComponent(Env_over.n_features, 
                              Env_over.init_weights, 
                              config_re["critic_structure"][1][0], 
                              dropout_prob,
                              "critic")
    Env_over.generator.grow(config_re["gen_structure"][1][1])
    Env_over.generator.grow(config_re["gen_structure"][1][2])
    
    Env_over.critic.grow(config_re["critic_structure"][1][1])
    Env_over.critic.grow(config_re["critic_structure"][1][2])
                                       
    if path_ID[-1] != 'e':
        telapsed_summary = Env_over.tuneAndTrain()
    else:
        telapsed_summary = Env_over.tuneAndTrain(preTrained=False)
    #save times
    telapsed_summary.to_csv(Env_over.path_results + "/time_training.csv")

    #############################################################################################################
    #Load underrep GAN for validation
    pre_generator = Generator(Env_under.n_features, 
                              Env_under.init_weights, 
                              config_re["gen_structure"][0][0], 
                              dropout_prob)
    pre_generator.grow(config_re["gen_structure"][0][1])
    pre_generator.grow(config_re["gen_structure"][0][2])

    pre_generator.load_pars(Env_under.path_pars, 
                            f"{pre_generator.n_pre_layers}Retraining", 
                            Env_under.device)
    pre_generator.change_dropout(dropout_prob)

    pre_generator.eval()
    to_device(pre_generator, Env_under.device)


    #Load overrep GAN for validation
    pre_generator2 = Generator(Env_over.n_features, 
                               Env_over.init_weights, 
                               config_re["gen_structure"][1][0], 
                               dropout_prob)
    pre_generator2.grow(config_re["gen_structure"][1][1])
    pre_generator2.grow(config_re["gen_structure"][1][2])

    pre_generator2.load_pars(Env_over.path_pars, 
                             f"{pre_generator2.n_pre_layers}Retraining", 
                             Env_over.device)
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
