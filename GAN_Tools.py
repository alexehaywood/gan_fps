#%%
from torch import device, cuda, load, mean, save, randn, linalg, from_numpy, float32, log, rand, square, ones_like, mm, zeros, exp, t, manual_seed
import torch.nn as nn
import random
from torch.autograd import grad

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from scipy.stats import skewtest

from os.path import exists
import os, shutil

from pickle import dump

from plotnine import ggplot, aes, geom_line, geom_vline, labels, scales, geom_segment, labs, scale_color_manual, theme, geom_text

from tqdm import tqdm

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, silhouette_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
import matplotlib.pyplot as plt
from statistics import fmean, stdev

from kneed import KneeLocator
from scipy.optimize import curve_fit

#%%
##Move tensors to GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if cuda.is_available():
        return device('cuda')
    else:
        return device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

#%%
##Define classes for training
class GanComponent(nn.Module):
    def __init__(self, n, init_weights_func, nodes_out, dropout_prob, component):
        super().__init__()
        self.component = component
        self.pre_network = nn.Sequential(
            nn.Linear(n, nodes_out),
            nn.SELU(),
            nn.AlphaDropout(p = dropout_prob),
            )
        
        self.structure = [nodes_out]

        self.pre_network.apply(init_weights_func)

        if self.component=="generator":
            self.layer_out = nn.Linear(nodes_out, n)
        elif self.component=="critic":
            self.layer_out = nn.Linear(nodes_out, 1)
        else:
            print("Error")
            
        nn.init.kaiming_normal_(self.layer_out.weight, nonlinearity='linear')
        self.layer_out.bias.data.fill_(0.01)

        self.pre_currentShape = nodes_out
        self.n_features_in = n
        self.n_pre_layers = 1

        self.dropout_prob = dropout_prob

    def grow(self, n):
        new_layer = nn.Linear(self.pre_currentShape, n)
        nn.init.kaiming_normal_(new_layer.weight, nonlinearity='linear')

        self.pre_network.add_module('grow_layer_'+str(n), new_layer)
        self.pre_network.add_module('selu_'+str(n), nn.SELU())
        self.pre_network.add_module('alpha_dropout_'+str(n), nn.AlphaDropout(p = self.dropout_prob))

        if self.component=="generator":
            self.layer_out = nn.Linear(n, self.n_features_in)
        elif self.component=="critic":
            self.layer_out = nn.Linear(n, 1)
        else:
            print("Error")
            
        nn.init.kaiming_normal_(self.layer_out.weight, nonlinearity='linear')
        self.layer_out.bias.data.fill_(0.01)

        self.n_pre_layers += 1
        self.structure.append(n)
        self.pre_currentShape = n

    def change_dropout(self, prob):
        self.dropout_prob = prob
        label = 'torch.nn.modules.dropout.AlphaDropout'
        for i in range(0, len(self.pre_network)):
            if type(self.pre_network[i] == label):
                self.pre_network[i].p = prob

    def forward(self, inp):
        out_pre = self.pre_network(inp)
        out = self.layer_out(out_pre)
        return(out)

    def save_pars(self, path, epoch_num):
        if self.component=="generator":
            save(self.state_dict(), path+'/pre_generator_pars_'+epoch_num+'.pt')
        elif self.component=="critic":
            save(self.state_dict(), path+'/pre_critic_pars_'+epoch_num+'.pt')
        else:
            print("Error")

    def load_pars(self, path, epoch_num, device):
        if self.component=="generator":
            self.load_state_dict(load(path+'/pre_generator_pars_'+epoch_num+'.pt', map_location=device))
        elif self.component=="critic":
            self.load_state_dict(load(path+'/pre_critic_pars_'+epoch_num+'.pt', map_location=device))
        else:
            print("Error")



class Results():
    def __init__(self, saved_results=None, path = None):
        """
        Initialise record.
        Saved_results allows user to generate new record, even if record file is present.
        """
        if saved_results is None:
            saved_results = exists(path + '/saved_results.csv')

        if saved_results == True:
            self.record = pd.read_csv(path + '/saved_results.csv')
        else:
            self.record = pd.DataFrame(columns = ['epoch', 'n_pre_layers', 'loss_critic', 'pred_fake',
                                                  'pred_real', 'loss_gen', 'norm_gen'])


    def new_record(self, n_pre_layers, results, report_rate):
        if len(self.record.loc[self.record['n_pre_layers'] == n_pre_layers]) != 0:
            epoch = self.record.loc[self.record['n_pre_layers'] == n_pre_layers].iloc[-1, 0] + 1*report_rate #get the next epoch number
        else:
            epoch = report_rate

        new = pd.DataFrame(np.array([epoch, n_pre_layers] + results)).T
        new.columns = self.record.columns
        self.record = pd.concat([self.record, new], join = 'inner')

    def view(self):
        print(self.record)

    def get_top_epochs(self, n_pre_layers):
        x = self.record.loc[self.record['n_pre_layers'] == n_pre_layers]

        diffs_loss = pd.DataFrame(np.column_stack([
            list(x['epoch']),
            list(abs(x['loss_critic'] - x['loss_gen']))
            ]), columns=['epoch', 'diffs'])
        diffs_loss_top = diffs_loss.sort_values(by = ['diffs']).iloc[0:7]

        #plot
        plot = (
            ggplot(None)
            + geom_line(data = self.record, color = 'red', mapping = aes(y = 'loss_critic', x = 'epoch'))
            + geom_line(data = self.record, color = 'blue', mapping = aes(y = 'loss_gen', x = 'epoch'))
            + geom_vline(xintercept = diffs_loss_top['epoch'], size = 0.2)
            + labels.xlab('Epoch')
            + labels.ylab('Loss')
            + scales.ylim(-10, 15)
        )
        return(diffs_loss.sort_values(by = ['diffs']), plot)

    def save_results(self, path):
        self.record.to_csv(path + '/saved_results.csv')
        
        
#%%
        
class Environment():
    def __init__(self, results_record=None, generator=None, critic=None,
          batch_size=None, path_dat=None, n_features=None, iter_critic=None, lr=None, alpha=None, func_optim=None, device=None, 
          transformer=None, path_results=None, beta=None, n_training_samples=None, path_pars=None, PATH=None, tune_params=None, path_prePars=None):
        """
        """
        self.results_record, self.generator, self.critic, self.batch_size, self.path_dat, self.n_features, self.iter_critic, self.lr, self.alpha, self.func_optim, self.device, self.transformer, self.path_results, self.beta, self.n_training_samples, self.path_pars, self.PATH, self.tune_params, self.path_prePars = results_record, generator, critic, batch_size, path_dat, n_features, iter_critic, lr, alpha, func_optim, device, transformer, path_results, beta, n_training_samples, path_pars, PATH, tune_params, path_prePars
    
        def init_weights(net):
            if type(net) == nn.Linear:
                nn.init.kaiming_normal_(net.weight, nonlinearity='linear')   
    
        self.init_weights = init_weights
        
    
    def findApEl(self, offset = 1, rate_save = 20):
        def objective(x, a, b, c, d, e, f, g, h, i, j, k):
        	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f * x**6) + (g * x**7) + h#(h * x**8) + (i * x**9) + (j * x**10) + k
        
        dat_x = self.results_record.record['epoch'].iloc[:offset-1]
        dat_y = self.results_record.record['loss_critic'].iloc[:offset-1]
        
        ##curve fit
        popt, _ = curve_fit(objective, dat_x, dat_y)
        # summarize the parameter values
        a, b, c, d, e, f, g, h, i, j, k = popt
        
        dat_obj = np.arange(min(dat_x), max(dat_x), 1)
        dat_obj = pd.DataFrame(
            {'x_line' : dat_obj,
            'y_line' : objective(dat_obj, a, b, c, d, e, f, g, h, i, j, k)}
            )
        
        apEl = KneeLocator(dat_obj['x_line'], dat_obj['y_line'], direction = 'decreasing')
        
        #plot
        plot = (
            ggplot(None)
            + geom_line(data = dat_obj, color = 'blue', mapping = aes(y = 'y_line', x = 'x_line'))
            + geom_vline(xintercept = apEl.knee, size = 0.2)
            + labels.xlab('Epoch')
            + labels.ylab('Loss')
        )
        return(int( rate_save * round(apEl.knee / rate_save) ), plot)
        
    def optimise_critic(self, inp_noise, batch_real):
        optimiser = self.func_optim(self.critic.parameters(), self.lr)
    
        batch_fake = self.generator(inp_noise)
    
        pred_fake = self.critic(batch_fake)
        pred_fake_clone = pred_fake.detach().clone() #Make a clone to pass to retain gradients and pass to the optimise_gen function
        pred_real = self.critic(batch_real)
    
        #Wasserstein distance
        loss = mean(pred_real) - mean(pred_fake)
    
        #gradient penalty
        unif = to_device(rand(self.batch_size, 1), self.device)
        interpolates = (unif * batch_real) + ((1-unif) * batch_fake)
        pred_interpolates = self.critic(interpolates)
        gradients = grad(outputs = pred_interpolates, inputs = interpolates,
                         grad_outputs= ones_like(pred_interpolates)) #check
        slopes = linalg.norm(gradients[0], ord = 2, dim = 0)
        grad_pen = mean( square(slopes - 1) )
    
        loss += self.beta * grad_pen
    
        #Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
        return(pred_fake_clone, mean(pred_real).item(), loss.item())
    
    
    def optimise_generator(self, pred_fake, n_sim):
        optimiser = self.func_optim(self.generator.parameters(), self.lr)
    
        #metric for similarity
        inp_noise = randn(n_sim, self.n_features)
        inp_noise = inp_noise.to(self.device)
    
        samps = self.generator(inp_noise)
        norm = linalg.matrix_norm(samps, ord = 2) #L2 matrix norm; greater value means less similar samples generated
    
        loss = mean(pred_fake) - self.alpha * log(norm) 
    
        #Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
        return(loss.item(), norm.item())
    
    
    def get_scaler(self, n_samples, name_transformer):
        """
        Returns a fitted ColumnTransformer
        """
        n = sum(1 for line in open(self.path_dat)) - 1 #number of records in file (excludes header)
        skip = sorted(random.sample(range(1,n+1),n-n_samples)) #the 0-indexed header will not be included in the skip list
        dat = pd.read_csv(self.path_dat, skiprows=skip)
        
        ind_skewCols = []
        ind_normalCols = []
    
        #check for skewed columns (sample size >= 8 needed)
        test_skew = dat.apply(skewtest, axis = 0)
        for i in range(0, len(test_skew.columns)):
            if test_skew.iloc[1,i] < 0.05:
                ind_skewCols.append(i)
            else:
                ind_normalCols.append(i)
    
        skewed_transformer = Pipeline(steps = [
            ('scaler', PowerTransformer(method = 'yeo-johnson')) #allows for -'ve values
            ])
        normal_transformer = Pipeline(steps = [
            ('scaler', RobustScaler())
            ])
    
        colTrans_early = ColumnTransformer(transformers=[
            ('skewed_trans', skewed_transformer, ind_skewCols),
            ('normal_trans', normal_transformer, ind_normalCols)
            ])
    
        ext_transformer = colTrans_early.fit(dat)
        dump(ext_transformer, open(self.path_results+name_transformer, 'wb')) #save transformer
        
        return(ext_transformer)
        
    
    def get_batch(self):
        #https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-python-data-frame
        n = sum(1 for line in open(self.path_dat)) - 1 #number of records in file (excludes header)
        skip = sorted(random.sample(range(1,n+1),n-self.batch_size)) #the 0-indexed header will not be included in the skip list
        dat = pd.read_csv(self.path_dat, skiprows=skip)
    
        dat = self.transformer.transform(dat)
    
        return(dat)
    
    def func_epoch(self, return_loss):
    
        for i in range(1, self.iter_critic):
            #get data ready
            batch_real = from_numpy(self.get_batch())
            batch_real = batch_real.to(float32)
            inp_noise = randn(self.batch_size, self.n_features, dtype=float32)
    
            batch_real = batch_real.to(self.device)
            inp_noise = inp_noise.to(self.device)
    
            #train critic
            pred_fake, pred_real, loss_critic = self.optimise_critic(inp_noise, batch_real)
        #train generator
        loss_gen, norm_gen = self.optimise_generator(pred_fake, 20)
    
        if return_loss == True:
            return([loss_critic, mean(pred_fake).item(), pred_real, loss_gen, norm_gen])
        else:
            return(None)
    
    
    def train(self, n_epochs, verbose=True):
    
        report_rate = 20
    
        if len(self.results_record.record.loc[self.results_record.record['n_pre_layers'] == self.generator.n_pre_layers]) != 0:
            start_epoch = self.results_record.record.loc[self.results_record.record['n_pre_layers'] ==
                                                    self.generator.n_pre_layers].iloc[-1, 0] + 1 #get the next epoch number
            start_epoch = int(start_epoch)
        else:
            start_epoch = 1
    
        iterator = range(start_epoch, start_epoch + n_epochs)
        if verbose == True:
            iterator_tqdm = tqdm(iterator)
        else:
            iterator_tqdm = iterator
        
        for i in ( iterator_tqdm ):
            if i % report_rate == 0:
                return_loss = True
                results = self.func_epoch(return_loss)
    
                self.results_record.new_record(self.generator.n_pre_layers, results, report_rate)
    
                self.critic.save_pars(self.path_pars, str(i))
                self.generator.save_pars(self.path_pars, str(i))
                self.results_record.save_results(self.path_results)
    
            else:
                return_loss = False
                results = self.func_epoch(return_loss)
        
        if verbose == True:
            prog = str(iterator_tqdm)
            time_elapsed = str(prog)[str(prog).find('[')+1:str(prog).find('<')].strip(' ')
            return(time_elapsed)
        
    
    def select_pars(self, epoch, remove_old = True):
        self.generator.load_pars(self.path_pars, epoch, self.device)
        self.critic.load_pars(self.path_pars, epoch, self.device)
    
        if remove_old == True:
            #https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
            folder = self.path_pars
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
                    
    def load_pre(self, dropout_prob, seed, preTrained = True, path_prePars = None):
        np.random.seed(seed)
        manual_seed(seed)
        random.seed(seed)
        
        flag_loadPars = True
        if preTrained:
            id_label = f"{self.generator.n_pre_layers}Pretraining"
        else:
            if self.generator.n_pre_layers-1 == 0:
                flag_loadPars = False
            else:
                id_label = f"{self.generator.n_pre_layers-1}Pretraining"
        
        struct_gen = self.generator.structure
        struct_critic = self.critic.structure
        self.generator = GanComponent(self.n_features, 
                                      self.init_weights, 
                                      struct_gen[0], 
                                      dropout_prob,
                                     "generator")
        self.critic = GanComponent(self.n_features, 
                                   self.init_weights, 
                                   struct_critic[0], 
                                   dropout_prob,
                                   "critic")
        range_upper = len(struct_gen)
        if not preTrained:
            range_upper-=1
        for i in range(1, range_upper):
            self.generator.grow(struct_gen[i])
            self.critic.grow(struct_critic[i])

        if flag_loadPars:
            if not preTrained:
                self.generator.load_pars(self.path_pars, f"complete_loop_{id_label}", self.device)
                self.critic.load_pars(self.path_pars, f"complete_loop_{id_label}", self.device)   
                
                self.generator.grow(struct_gen[-1])
                self.critic.grow(struct_critic[-1])
            else:
                self.generator.load_pars(path_prePars, f"complete_loop_{id_label}", self.device)
                self.critic.load_pars(path_prePars, f"complete_loop_{id_label}", self.device)   
        
        self.generator.train()
        self.critic.train()
        
        to_device(self.generator, self.device)
        to_device(self.critic, self.device)
        
        
    def auto_tune(self, max_epochs_1, max_loss_1, n_epochs_2, dropout_prob, seed, 
                  lr_1, lr_2,
                  rate_save = 20, diff_epochs = 500, max_loss_2 = 15, min_loss_2 = -10,
                  alpha_instab = 3, preTrained = True):
        results_tuning_1 = {
            'lr_1' : [],
            'max_epochs_1' : [],
            'final_loss' : [],
            'max_loss' : [],
            'time_taken' : [],
            'epoch_final' : [],
            'final_select' : None
            }
        results_tuning_2 = {
            'lr_2' : [],
            'n_epochs_2' : [],
            'max_loss_2' : [],
            'min_loss_2' : [],
            'ind_null' : [],
            'instab' : [],
            'diff' : [],
            'time_taken' : [],
            'final_lr' : None
            }
        
        final_epochs = []
        final_losses = []
        for i in range(0, len(lr_1), 1):
            lr = lr_1[i]
            self.load_pre(dropout_prob, 118, preTrained = preTrained, path_prePars=self.path_prePars)
            self.results_record = Results(path = self.path_results, saved_results = False)
            
            iterator_tqdm = tqdm(range(100, max_epochs_1 + 100, 100), desc = 'lr_1 %i out of %i' %  (i+1, len(lr_1)) )
            for n_epochs in iterator_tqdm:
                self.lr = lr
                self.train(100, verbose=False)
                
                losses = self.results_record.record['loss_critic']
                
                if losses.iloc[-1] < max_loss_1:
                    if losses.iloc[0] < max_loss_1:
                        loss_final = losses.iloc[0]
                        epoch_final = 0
                    else:
                        ind_final = losses.lt(max_loss_1).argmax() -1

                        loss_final = losses.iloc[ind_final]
                        epoch_final = self.results_record.record['epoch'].iloc[ind_final]

                    print('Final loss reached: {}'.format(loss_final))
                    break
                
                if n_epochs == max_epochs_1:
                    loss_final = losses.iloc[-1]
                    epoch_final = max_epochs_1
                    
                    print('Final loss reached: {}'.format(loss_final))
                
            prog = iterator_tqdm
            time = str(prog)[str(prog).find('[')+1:str(prog).find('<')].strip(' ')
            
            final_epochs.append(epoch_final)
            final_losses.append(loss_final)
            
            results_tuning_1['lr_1'].append(lr)
            results_tuning_1['max_epochs_1'].append(max_epochs_1)
            results_tuning_1['epoch_final'].append(epoch_final)
            results_tuning_1['final_loss'].append(loss_final)
            results_tuning_1['max_loss'].append(max_loss_1)
            results_tuning_1['time_taken'].append(time)
            
            iterator_tqdm.update()
            
        i = final_losses.index(min(final_losses))
        
        final_lr = [0] * len(lr_1)
        final_lr[i] = 1
        results_tuning_1['final_select'] = final_lr

        lr_1 = lr_1[i]
        n_epochs_1 = np.int64(final_epochs[i])


        final_instabs = []
        final_diffs = []
        for i in range(0, len(lr_2)):
            lr = lr_2[i]
            print( 'lr_2 %i out of %i' %  (i+1, len(lr_2)) )
            
            self.load_pre(dropout_prob, 118, preTrained = preTrained, path_prePars=self.path_prePars)
            self.results_record = Results(path = self.path_results, saved_results = False)

            #1st round of training
            if n_epochs_1 != 0:
                self.lr = lr_1    
                telapsed_1 = self.train(n_epochs_1)
            else:
                telapsed_1 = pd.NA
            
            #2nd round of training
            self.lr = lr
            telapsed_2 = self.train(n_epochs_2)
            
            
            losses = self.results_record.record['loss_critic'].iloc[-int(n_epochs_2/rate_save):]     
            
            ind_check = np.array([np.nan, np.nan])
            
            if losses.gt(max_loss_2).sum() > 0: #check for null values
                ind_check[0] = losses.gt(max_loss_2).argmax()
            if losses.lt(min_loss_2).sum() > 0:
                ind_check[1] = losses.lt(min_loss_2).argmax()
            
            if np.isnan(ind_check).sum() == 2: #no values gt or lt limits
                ind_null = len(losses)
            else:
                ind_null = int(ind_check[np.nanargmin(ind_check)]) + 1
                
            losses = losses[:ind_null]
            print('Null value at index {}'.format(ind_null))
            
            apex = self.get_apex(losses.tolist())
            if len(apex)<50:
                top_x_diffs=len(apex)
            else:
                top_x_diffs=50
            instab, _ = self.get_instab(apex, top_x_diffs)
            
            plot_losses = (
                ggplot(self.results_record.record.iloc[:int(n_epochs_1/rate_save) + ind_null])
                + geom_line(color = 'red', mapping = aes(y = 'loss_critic', x = 'epoch'))
                + geom_line(color = 'blue', mapping = aes(y = 'loss_gen', x = 'epoch'))
                + labels.xlab('Epoch')
                + labels.ylab('Loss')
                + scales.ylim(-10, 15)
            )
            
            plot_losses.save(self.path_results+'/{} tuning_loss'.format(lr) + '_retrain' + '.png')
            
            diff = losses.iloc[:int(diff_epochs/rate_save)].mean() - losses.iloc[-int(diff_epochs /rate_save) -ind_null:].mean()
            
            print('Final instab and diff: {}, {}'.format(instab, diff))
            
            final_instabs.append(instab)
            final_diffs.append(diff)
            
            results_tuning_2['lr_2'].append(lr)
            results_tuning_2['n_epochs_2'].append(n_epochs_2)
            results_tuning_2['max_loss_2'].append(max_loss_2)
            results_tuning_2['min_loss_2'].append(min_loss_2)
            results_tuning_2['ind_null'].append(ind_null)
            results_tuning_2['instab'].append(instab)
            results_tuning_2['diff'].append(diff)
            results_tuning_2['time_taken'].append([telapsed_1, telapsed_2])


        best = [ element1 - element2 for (element1, element2) in zip([n*alpha_instab for n in final_instabs], final_diffs) ]
        i = best.index(min(best))

        final_lr = [0] * len(lr_2)
        final_lr[i] = 1
        results_tuning_2['final_lr'] = final_lr

        lr_2 = lr_2[i]
        
        return(lr_1, n_epochs_1, lr_2, results_tuning_1, results_tuning_2)
    
    def tuneAndTrain(self, preTrained=True):
        """
        ...
        """
        max_loss = self.tune_params["max_loss_2"] 
        min_loss = self.tune_params["min_loss_2"]
        
        lr_1, n_epochs_1, lr_2, results_tuning_1, results_tuning_2 = \
            self.auto_tune(**self.tune_params, preTrained=preTrained)

        results_tuning_1 = pd.DataFrame.from_dict(results_tuning_1)
        results_tuning_2 = pd.DataFrame.from_dict(results_tuning_2)
        
        if preTrained:
            id_label = f"{self.generator.n_pre_layers}Retraining"
        else:
            id_label = f"{self.generator.n_pre_layers}Pretraining"
        results_tuning_1.to_csv(self.path_results + f"/results_tuning_1_{id_label}.csv")
        results_tuning_2.to_csv(self.path_results + f"/results_tuning_2_{id_label}csv")

        print('Training')
        self.lr = lr_1
        telapsed_1 = self.train(n_epochs_1)

        self.lr = lr_2
        telapsed_2 = self.train(10000-n_epochs_1)

        #save times
        telapsed_summary = pd.DataFrame([ telapsed_1, telapsed_2 ])

        ##Grow
        #input
        ##check for null values
        i_losses = self.results_record.record['n_pre_layers'] == self.generator.n_pre_layers
        losses = self.results_record.record['loss_critic'][i_losses]
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

        nEpoch_pars, graph_loss = self.findApEl(offset = ind_null)

        plot = (
            ggplot(self.results_record.record.iloc[:ind_null-1])
            + geom_line(color = 'red', mapping = aes(y = 'loss_critic', x = 'epoch'))
            + geom_line(color = 'blue', mapping = aes(y = 'loss_gen', x = 'epoch'))
            + geom_vline(xintercept = nEpoch_pars, size = 0.2)
            + labels.xlab('Epoch')
            + labels.ylab('Loss')
            + scales.ylim(-10, 15)
        )

        graph_loss.save(self.path_results+f"/plot_loss_bestFit_loop_{id_label}.png")
        plot.save(self.path_results+f"/plot_loss_loop_{id_label}.png")
        self.select_pars(str(nEpoch_pars))
        self.generator.save_pars(self.path_pars, f"complete_loop_{id_label}")
        self.critic.save_pars(self.path_pars, f"complete_loop_{id_label}")

        self.results_record.save_results(self.PATH + '/Results')

        return telapsed_summary

    
        
    def get_apex(self, dat_y):
        apex = []

        #check if at current point wave is increasing/decreasing
        if dat_y[0] > dat_y[1]:
            decreasing = True
        else:
            decreasing = False

        for i in range(1, len(dat_y), 1):

            if decreasing == True:
                if dat_y[i-1] < dat_y[i]:
                    apex.append(dat_y[i-1])

                    decreasing = False

            if decreasing == False:
                if dat_y[i-1] > dat_y[i]:
                    apex.append(dat_y[i-1])

                    decreasing = True

        return(apex)            

    def get_instab(self, dat_y, top_x_diffs = 50):
        compare_i1 = list( range(0, len(dat_y)-1, 1) )
        compare_i2 = list( range(1, len(dat_y), 1) )

        diffs = np.absolute(np.array(dat_y)[compare_i1] - np.array(dat_y)[compare_i2])
        diffs = np.sort(diffs)[-top_x_diffs:]

        return(np.mean(diffs), diffs)



#%%
##Validation Function
def classification_metrics(path_dat_train, path_dat_val, generator, path_transformer, n_features, device, PATH,
                           generator2 = None, expand_size = None):
    
    transformer = joblib.load(path_transformer)

    dat_x = pd.read_csv(path_dat_train)
    dat_y = dat_x.iloc[:, 0]
    label_encod = LabelEncoder().fit(dat_y)
    dat_y = label_encod.transform(dat_y)
    dat_x = dat_x.iloc[:, 1:len(dat_x.columns)+1]
    dat_x = pd.DataFrame(transformer.transform(dat_x))

    dat_val_x = pd.read_csv(path_dat_val)
    dat_val_y = dat_val_x.iloc[:, 0]
    dat_val_y = label_encod.transform(dat_val_y)
    dat_val_x = dat_val_x.iloc[:, 1:len(dat_val_x.columns)+1]
    dat_val_x = pd.DataFrame(transformer.transform(dat_val_x))

    for i_classifier in range(0,3):
        results_classification = pd.DataFrame(None,
            columns=['impute_method', 'mean_roc_auc', 'mean_accuracy', 'hyp'])

        summary_text = pd.DataFrame(columns = ['summary', 'pos', 'col'], index = [1, 2, 3, 4])
        plot = ( 
            ggplot()
            + labels.xlab('Mean FPR')
            + labels.ylab('Mean TPR')
            + scales.ylim(-0.05, 1.05)
            + scales.xlim(-0.05, 1.05)
            + scale_color_manual(values = ['m', 'b', 'g', 'y'], limits = ['ExpandGAN', 'GAN', 'SMOTE', 'RO']) 
           )
        
        
        for i_methods in tqdm(range(0,4)):
            if generator2 is not None:
                impute_method = 'ExpandGAN'
                col = 'm'
                def impute_dat():
                    n_impute = sum(dat_y == 1) - sum(dat_y == 0)
                    n_expand = sum(dat_y == 1)

                    #underrep class
                    noise_inp = randn(n_impute + n_expand, n_features)
                    noise_inp = noise_inp.to(device)
                    to_device(generator, device)

                    dat_impute = generator(noise_inp)
                    dat_impute = dat_impute.to('cpu')
                    dat_impute = pd.DataFrame(dat_impute.detach().numpy())

                    dat_x_imp = pd.concat([dat_x, dat_impute], join = 'inner')
                    dat_y_imp = np.concatenate([dat_y, [0]*(n_impute+n_expand)])

                    #overrep class
                    noise_inp = randn(n_expand, n_features)
                    noise_inp = noise_inp.to(device)
                    to_device(generator2, device)

                    dat_impute = generator2(noise_inp)
                    dat_impute = dat_impute.to('cpu')
                    dat_impute = pd.DataFrame(dat_impute.detach().numpy())

                    dat_x_imp = pd.concat([dat_x_imp, dat_impute], join = 'inner')
                    dat_y_imp = np.concatenate([dat_y_imp, [1]*n_expand])

                    return(dat_x_imp, dat_y_imp)

                dat_x_imp, dat_y_imp = impute_dat()

            else:
                continue

        if i_methods == 1:
            #GAN
            impute_method = 'GAN'
            col = 'b'

            def impute_dat():
                n_impute = sum(dat_y == 1) - sum(dat_y == 0)

                noise_inp = randn(n_impute, n_features)
                noise_inp = noise_inp.to(device)
                to_device(generator, device)

                dat_impute = generator(noise_inp)
                dat_impute = dat_impute.to('cpu')
                dat_impute = pd.DataFrame(dat_impute.detach().numpy())

                dat_x_imp = pd.concat([dat_x, dat_impute], join = 'inner')
                dat_y_imp = np.concatenate([dat_y, [0]*n_impute])
                return(dat_x_imp, dat_y_imp)

            dat_x_imp, dat_y_imp = impute_dat()

        if i_methods == 2:
            impute_method = 'SMOTE'
            col = 'g'

            if sum(dat_y == 0) < 6:
                k_neighbors = sum(dat_y == 0) -1
            else:
                k_neighbors = 5

                def impute_dat():
                    dat_x_imp, dat_y_imp = SMOTE(k_neighbors = k_neighbors).fit_resample(dat_x, dat_y)
                    return(dat_x_imp, dat_y_imp)

            dat_x_imp, dat_y_imp = impute_dat()

        if i_methods == 3:
            impute_method = 'RO'
            col = 'y'

            def impute_dat():
                dat_x_imp, dat_y_imp = RandomOverSampler().fit_resample(dat_x, dat_y)
                return(dat_x_imp, dat_y_imp)

            dat_x_imp, dat_y_imp = impute_dat()

        #check
        print( ' %s : %i samples class_1; %i samples class_2' %(impute_method, sum(dat_y_imp == 0), sum(dat_y_imp == 1)) )

        #Make classification model
        """
        Find optimum hyperparameters
        Then, instantiate a new classifier with opt_hyps followed by predicition metrics
            Repeat and take avgs
        """
        if i_classifier == 0:
            classifier = 'histgradboost'
            model_val = HistGradientBoostingClassifier()
            pars = {
                'learning_rate' : [0.001, 0.01, 0.1, 1, 10],
                'min_samples_leaf' : [1, 10, 25, 50, 75]
                }
        if i_classifier == 1:
            classifier = 'svm'
            model_val = SVC()
            pars = {
                'C' : [1, 5, 10, 25, 50, 75],
                'gamma' : [0.1, 0.4, 0.8]
                }
        if i_classifier == 2:
            classifier = 'elasticNet'
            model_val = ElasticNet()
            pars = {
                'alpha' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
                'l1_ratio' : [0, 1, 0.01],
                'max_iter' : [100000]
                }

        scorer = make_scorer(roc_auc_score)

        scores_roc = []
        scores_acc = []
        scores_interpTpr = []

        mean_fpr = np.linspace(0, 1, 100)

        cv = StratifiedKFold(n_splits = 5, shuffle = True)
        searcher = GridSearchCV(model_val, pars, scoring = scorer, n_jobs = -1, cv = cv, refit=True).fit(dat_x_imp, dat_y_imp)

        #Due to randomness of model method, build model n times and impute data n times and take avg performance
        for i2 in range(1, 100):
            if i_classifier == 0:
                model_val = HistGradientBoostingClassifier(**searcher.best_params_)
            if i_classifier == 1:
                model_val = SVC(**searcher.best_params_)
            if i_classifier == 2:
                model_val = ElasticNet(**searcher.best_params_)
                
            dat_x_imp, dat_y_imp = impute_dat()
            model_val = model_val.fit(dat_x_imp, dat_y_imp)

            preds = model_val.predict(dat_val_x)

            scores_roc.append(roc_auc_score(dat_val_y, preds))
            scores_acc.append(accuracy_score(dat_val_y, preds))


            fpr, tpr, thresholds = roc_curve(dat_val_y, preds)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            scores_interpTpr.append(interp_tpr)

        results_entry = pd.DataFrame(np.column_stack(
            [impute_method, fmean(scores_roc), fmean(scores_acc),
             str(searcher.best_estimator_)],
            ),
 
            
        columns=['impute_method', 'mean_roc_auc', 'mean_accuracy', 'hyp'])
        results_classification = pd.concat([results_classification, results_entry], join='inner')

        #roc plot info
        mean_tpr = np.mean(scores_interpTpr, axis=0)
        mean_tpr[-1] = 1.0


        dat_plot = pd.DataFrame.from_dict({
            'mean_fpr' : mean_fpr,
            'mean_tpr' : mean_tpr,
            'group' : [impute_method] * len(mean_fpr)
            })    
        plot = plot + geom_line(data = dat_plot, mapping = aes(x = 'mean_fpr', y = 'mean_tpr', color = 'group'), size = 1)
            
        summary_text.iloc[i_methods, 0] = r"%s (AUC = %0.2f +- %0.2f)" % (impute_method, fmean(scores_roc), stdev(scores_roc))
        summary_text.iloc[3-i_methods, 1] = i_methods * 0.1
        summary_text.iloc[i_methods, 2] = impute_method
        


    plot = plot + geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), colour = 'r', linetype = 'dashed', alpha = 0.8, size = 1)
    plot = plot + labels.ggtitle("Receiver Operating Characteristic: " + classifier) + labs(color = 'Legend') + \
        theme(legend_position = 'right', legend_direction='vertical') + \
        geom_text(data = summary_text, mapping = aes(label = 'summary', y = tuple(summary_text['pos']),\
                                                     x = 0.7, color = tuple(summary_text['col'])))
    
    plot.save(PATH + '/Results/' + classifier + '.png')
    results_classification.to_csv(PATH + '/Results/' + classifier + '.csv')
