from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from optuna import distributions
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.samplers._random import RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import TrialState
from optuna.samplers._base import _process_constraints_after_trial

import copy, time, ctypes, os, subprocess
import numpy as np
from scipy.stats import norm

import torch, multiprocessing
from optuna.samplers._hyperjump.gps import ExactGPModel, EasyGPModel
from scipy import LowLevelCallable
from scipy.integrate import quad, nquad
from mpmath import mp

#gcc -shared -fPIC -o func.so func.c
#path_to_risk_c = './optuna/samplers/_hyperjump/func.so'
path_to_risk_c = os.path.dirname(os.path.abspath(__file__))
path_to_risk_c = path_to_risk_c.replace('_hyperjump.py', '')
if not os.path.isfile(path_to_risk_c + '/func.so'):
    print('Generating c file to compute the risk...')
    subprocess.run('gcc -shared -fPIC -o ' + path_to_risk_c + '/func.so ' + path_to_risk_c + '/func.c', shell=True,)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class HyperJump(BaseSampler):
    """Sampler using HyperJump.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    Args:
        eta: sucessive halving reduction factor
        min_budget: minimum budget
        max_budget: maximum budget
        budgetName: name of the budget dimension to include in the dict (e.g., epochs, budget)
        seed: Seed for random number generator.
    """
     
    def __init__(self, 
                    eta: int = 3, 
                    min_budget: float = 1.0,
                    max_budget: float = 81.0,
                    budgetName: str = "budget",
                    random_fraction: float = 0.3,
                    num_samples: int = 100,
                    threshold: float = 0.1,
                    random_fraction_to_not_jump: float = 0.1,
                    seed: Optional[int] = None, 
                    pool = None) -> None:
        
        self.prints = False #True

        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self._search_space = IntersectionSearchSpace()
        self.budgetName = budgetName

        self.configQueued = []  # configs scheduled to be run
        self.configRun = [] #all configs that were run in the curretn bracket
        self.configRunAll = [] # all configs that were run

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.allBudgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))
        #self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))

        self.currentIteration = -1 # this value is incremente in nextIteration
        self.nextIteration() # update self.budgets and self.num_configs

        self.model2Use = 'EasyGPModel' #'ExactGPModel'

        self.model = None
        self.modelIsTrained = False
        self.x = None  
        self.y = None

        self.min_number_sample_model = 0 #10
        self.random_fraction = random_fraction
        np.random.seed(seed)
        self.num_samples = num_samples

        self.thresholdRisk = threshold # threshold to decode whether to jump or not 
        self.random_fraction_to_not_jump = random_fraction_to_not_jump

        self.pool = None #pool #multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) #



    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict


    def reseed_rng(self) -> None:
        self._rng.rng.seed()


    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        search_space: Dict[str, BaseDistribution] = {}

        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space


    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        
        
        Sel, Unsel = self.analyse_risk(study, trial)
        config = self.runNextConfig(study,trial, Sel, Unsel)
        if self.prints: print("sample config " + str(config) + '\n')
        if config == {}: 
            return {}
        elif config is not None:
            config[self.budgetName] = self.budgets[self.stage]
            return config
        

        if (self.actual_num_configs[self.stage] < self.num_configs[self.stage]):
            # queue is empty, so add all the configs for the current stage to the queue
            # we need to add all the configs at the begining of each stage in order to compute the
            #       selected and discarded sets and then compute the risk
            while (self.actual_num_configs[self.stage] < self.num_configs[self.stage]):
                config =  self._sample_relative(study, trial, search_space,)

                self.actual_num_configs[self.stage] += 1
                budget = self.budgets[self.stage]
                self._queue(config, budget) 

                if config == {}: break
                    # just run the first time, when we don't have a search space
                    #self.num_running += 1
                    #return {}

                config[self.budgetName] = budget
            if self.prints:  print('stage ' + str(self.stage) + ' with ' + str(len(self.configQueued)) + ' configs.')
            return self.sample_relative(study,trial,search_space)

        #if it arrieves here, all configs were queued and are running. 
        # there are no more configs to deploy
        # so advance to next stage (or next bracket)
        direction = "MAXIMIZE" if "MAXIMIZE" in str(study.direction) else 'MINIMIZE'
        self.nextStage(direction)
        return self.sample_relative(study,trial,search_space)


    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        if param_name==self.budgetName:
            return self.budgets[self.stage]
        
        search_space = {param_name: param_distribution}
        trans = _SearchSpaceTransform(search_space)
        trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
        val = trans.untransform(trans_params)[param_name]
        #self.config_sampleIndependent[param_name] = val # to be save in the future
        return val


    def _sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        if search_space == {}:
            return {}
        
        if not self.modelIsTrained and len(self.configRunAll) > self.min_number_sample_model:
            self.trainModel(study) # build the model

        if self.model is None or np.random.rand() < self.random_fraction:
            # sample at random
            return self._sample_random(search_space)
        
        #compute the BO EI to select config to sample
        return self.BO_EI(study, search_space)


    def _sample_random(self, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
            trans = _SearchSpaceTransform(search_space)
            trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
            return trans.untransform(trans_params)


    def _queue(self, toAdd, budget):
        self.configQueued.append((toAdd, budget))


    def _remove(self, config):
        tuple2remove = (config, config[self.budgetName])
        if tuple2remove in self.configQueued:
            self.configQueued.remove(tuple2remove)


    def _done(self, study: Study, trial: FrozenTrial, value: float):
        # Sort the keys alphabetically
        job = trial.params
        sorted_keys = sorted(job.keys())
        job_sorted = {key: job[key] for key in sorted_keys}
        
        budget = job_sorted[self.budgetName]
        for ct, bud in enumerate(self.budgets):
            if budget ==bud: 
                self.done_num_configs[ct] += 1
                break

        self.configRun.append({'config': job_sorted, 'value':value})
        self.configRunAll.append({'config': job_sorted, 'value':value, 'trial':trial})

        arrayConfig = self.config_to_array(job_sorted, study, trial)
        
        if self.model2Use == 'EasyGPModel':
            self.x = np.concatenate((self.x, np.array([arrayConfig])), axis=0) if self.x is not None else np.array([arrayConfig])
            #self.x = np.concatenate((self.x, np.array([list(job_sorted.values())])), axis=0) if self.x is not None else np.array([list(job_sorted.values())])
            self.y = np.concatenate((self.y, np.array([value])), axis=0) if self.y is not None else np.array([value])

        else:
            self.x = torch.cat((self.x, torch.tensor([arrayConfig])), dim=0) if self.x is not None else torch.tensor([arrayConfig])
            self.y = torch.cat((self.y, torch.tensor([value])), dim=0) if self.y is not None else torch.tensor([value])
        

    def successiveHalving(self, losses, direction):
        if direction == "MAXIMIZE":
            ranks = np.argsort(np.argsort(-np.array(losses)))
            #ret = ranks >= self.num_configs[self.stage]
        else:
            ranks = np.argsort(np.argsort(losses))
        ret = ranks < self.num_configs[self.stage]

        return ret
    

    def nextIteration(self): # same as nextBracket
        if self.prints: print('next iter')
        # number of 'SH rungs'
        self.currentIteration += 1
        s = self.max_SH_iter - 1 - (self.currentIteration%self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)

        self.num_configs = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        self.actual_num_configs = [0]*len(self.num_configs)
        self.done_num_configs = [0] * len(self.num_configs)
        self.budgets = self.allBudgets[(-s-1):]
        self.num_running = 0
        self.stage = 0
        self.configRun = []


    def nextStage(self, direction):
        if self.prints: print("nextStage - list configs queued " + str(self.configQueued))
        self.stage += 1
        if (self.stage >= len(self.num_configs)):
            # next iteration or bracket
            self.nextIteration()
            return

        budget = self.budgets[self.stage-1]

        #next stage
        losses = np.array([config['value'] for config in self.configRun if config['config'][self.budgetName]==budget])
        configs = [config['config'] for config in self.configRun if config['config'][self.budgetName]==budget]
        advance = self.successiveHalving(losses, direction)

        for i, config in enumerate(configs):
            if advance[i]:
                self.actual_num_configs[self.stage] += 1
                budget = self.budgets[self.stage]
                config[self.budgetName] = budget
                self._queue(copy.deepcopy(config), budget) 


    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        # here we want to remove configs from runningQueue to RunQueue
        # hyperband is a single-optimization problem - so we just care about the first value
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._done(study, trial, values[0])
        self.modelIsTrained = False


    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if len(self.configRunAll) == 0: # first time, so load data is exists
            no_hps_max = self.load_data(study)
            if no_hps_max > 0: self.min_number_sample_model = no_hps_max + 1
            # if the DB is empty, we only have info about no_hps_max after sample the first config
        
        
        if self.min_number_sample_model == 0:  # update the number of hyper-parameters 
            no_hps_max = self.count_HPs(study)
            if no_hps_max > 0: self.min_number_sample_model = no_hps_max + 1
                

    def load_data(self, study: Study, ) -> None:
        if self.prints: print("Loading data...")
        trials = study._get_trials(deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED])
        no_hps_max = 0
        for trial in trials:
            sorted_keys = sorted(trial.params.keys())
            no_hps = len(sorted_keys)
            job_sorted = {key: trial.params[key] for key in sorted_keys}
            self.configRunAll.append({'config': job_sorted, 'value': trial.value, 'trial': trial})

            if no_hps > no_hps_max: no_hps_max = no_hps
        return no_hps_max


    def count_HPs(self, study: Study, ):
        trials = study._get_trials(deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED])
        no_hps_max = 0
        for trial in trials:
            no_hps = len(trial.params.keys())
            if no_hps > no_hps_max: no_hps_max = no_hps
        return no_hps_max


    def trainModel(self, study: Study,) -> None:
        if self.prints: print("Train the model...")
        # Sort the keys alphabetically

        if self.model2Use == 'EasyGPModel':
            if self.model is None: self.model = EasyGPModel()

            if self.x is None:
                self.x =  np.array([self.config_to_array(config['config'], study, config['trial']) for config in self.configRunAll])
            if self.y is None:
                self.y = np.array([config['value'] for config in self.configRunAll])

        else:

            if self.x is None:
                self.x = torch.tensor([self.config_to_array(config['config'], study, config['trial']) for config in self.configRunAll])
            if self.y is None:
                self.y = torch.tensor([config['value'] for config in self.configRunAll])

            # initialize likelihood and model
            #if self.model is None:
            self.model = ExactGPModel(self.x, self.y) # create and train
            #self.model.train_gp() # just train

        self.model.train_gp(self.x, self.y)
        
        self.modelIsTrained = True


    def getBudgets(self, ):
        return self.allBudgets


    def predict(self, configs, study: Study, trial: FrozenTrial, budget=None):
        configs_predict = np.array([]) if self.model2Use == 'EasyGPModel' else torch.tensor([])


        for conf in configs:
            if budget is not None: conf[self.budgetName]=budget
        
            sorted_keys = sorted(conf.keys())
            conf_sorted = {key: conf[key] for key in sorted_keys}

            if self.model2Use == 'EasyGPModel':
                sample = np.array([self.config_to_array(conf_sorted, study, trial)])
                configs_predict = np.concatenate((configs_predict, sample), axis=0) if len(configs_predict)>0 else sample
            else:
                sample = torch.tensor([self.config_to_array(conf_sorted, study, trial)])
                configs_predict = torch.cat((configs_predict, sample), dim=0)

        values, stds = self.model.eval_gp(configs_predict)
        return configs, values, stds


    def BO_EI(self, study: Study, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
        #if self.prints: print("Computing the BO EI\n")
        if self.model2Use == 'EasyGPModel':
            configs_ei = np.array([]) # array of configs to compute ei
        else:
            configs_ei = torch.tensor([]) # tensor of configs to compute ei

        list_configs_ei = [] # list of configs(dict) to compute EI

        count_it = 0
        configs_tested = [conf['config'] for conf in self.configRunAll]

        no_hps_max = 0
        trial_max = None
        for trial in study.trials:
            no_hps = len(trial.params)
            if no_hps > no_hps_max:
                trial_max = trial
                no_hps_max = no_hps

        while len(configs_ei) < self.num_samples:
            # Filter the configurations that were already sampled
            sample = self._sample_random(search_space)
            sample[self.budgetName] = self.max_budget # if tested on full budget, we don´t want to insert
            if sample not in configs_tested and sample not in list_configs_ei:
                list_configs_ei.append(sample)

                if self.model2Use == 'EasyGPModel':
                    _sample = np.array([self.config_to_array(sample, study, trial_max)])
                    configs_ei = np.concatenate((configs_ei, _sample), axis=0) if len(configs_ei)>0 else _sample
                else:
                    _sample = torch.tensor([self.config_to_array(sample, study, trial_max)])
                    configs_ei = torch.cat((configs_ei, _sample), dim=0)
            
            count_it += 1
            if count_it == self.num_samples*10: break

        mean, var = self.model.eval_gp(configs_ei)

        mask = var==0  # Create a mask of zero values
        var[mask] = 10e-20  # Replace zero values with small_value
        if not isinstance(mean, np.ndarray): mean=mean.numpy()
        if not isinstance(var, np.ndarray): var=var.numpy()
       
        incumbent_value = study.best_value
        z = (incumbent_value - mean ) / var
        ei_value = var * (z * norm.cdf(z) + norm.pdf(z))

        idx =  np.argmax(ei_value)
        config  = list_configs_ei[idx]
        config[self.budgetName] = self.budgets[self.stage]
        return config
    

    def config_to_array(self, config, study: Study, trial: FrozenTrial):
        # convert a config dict into a numpy array to be used in the models

        if trial.distributions == {}:
            #we need this because in the first trial, this parameter is {}
            all_trials = study.trials
            if len(all_trials) == 0:  return []
            _trial = all_trials[0]
        else:
            _trial = trial

        _list_config = []
        for hp_name, val in config.items():
            if isinstance(val, str): # not numeric
                choices = list(_trial.distributions[hp_name].choices)
                it = 0
                for hp_val in choices:
                    if hp_val==val:
                        _list_config.append(it)
                        break
                    it += 1
            else:
                _list_config.append(val)

        return _list_config


    def analyse_risk(self, study: Study, trial: FrozenTrial):
        # compute the expected risk/loss reduction 
        # and if above a given threshold it jumps 
        direction = "MAXIMIZE" if "MAXIMIZE" in str(study.direction) else 'MINIMIZE'

        if not self.modelIsTrained and len(self.configRunAll) > self.min_number_sample_model:
            self.trainModel(study) # train the model
        else:
            return [],[]  # no model -> no risk computation

        if self.prints: print("Analyzing the risk of jumping...")

        #if in the last stage -> not run hyperjump
        if self.stage == len(self.num_configs)-1:
            untested_configs, untested_values, untested_std = self.get_untested_configs(study, trial)
            if not untested_configs:
                return [],[] #all configs were tested
              
            incumbent_conf, incumbent_value = self.get_incumbent(direction) 
            if incumbent_value is None: 
                #no incumbent - so, test the config
                return [],[] 

            SEL = [[incumbent_conf, incumbent_value, None]]
            UNSEL = []
            for i in range(0, len(untested_configs)):
                UNSEL.append([untested_configs[i], untested_values[i], untested_std[i]])


            EAR = self.risk(SEL, UNSEL, self.stage, direction, study, trial)[2] 
            if direction == 'MAXIMIZE':
                if incumbent_value==0: incumbent_value=0.01
                T = self.thresholdRisk / incumbent_value
            else:
                T = self.thresholdRisk * incumbent_value

            if EAR<T and np.random.uniform(0, 1) < 1-self.random_fraction_to_not_jump: 
                #jump to next bracker -> SEL = []  UNSEL = all_configs
                if self.prints: print("Jump to next bracket (with EAR of " + str(EAR) + ")" )

                self.process_results_to_jump(self.stage, [], UNSEL)
                SEL = []
                UNSEL = []
                
            return SEL, UNSEL


        # return the stage to jump or the current stage if not to jump
        tested_configs, tested_values = self.get_tested_configs()
        if not tested_configs: return [],[]
     
        untested_configs, untested_values, untested_std = self.get_untested_configs(study, trial)
        if not untested_configs: return [],[]

        all_configs = []     
        for i in range(0, len(tested_configs)):
            all_configs.append([tested_configs[i], tested_values[i], None])
        for i in range(0, len(untested_configs)):
            all_configs.append([untested_configs[i], untested_values[i], untested_std[i]])
        
        prevUNSEL_cumulative = []
        prevUNSEL = []
        prevSEL = []
        EAR = 0.0

        incumbent_conf, incumbent_value = self.get_incumbent(direction) 
        if incumbent_conf is None: incumbent_value = 0.01 if direction == 'MAXIMIZE' else 1.0 

        for targetStage in range (self.stage+1, len(self.num_configs)):
            m = self.num_configs[targetStage] #no of configs to test in next stage
            SEL, UNSEL = self.createSEL(all_configs, m, direction, study, trial)
            EAR += self.risk(SEL, UNSEL, targetStage-1, direction, study, trial)[2] 

            if direction == 'MAXIMIZE':
                if incumbent_value==0: incumbent_value=0.01
                T = self.thresholdRisk / incumbent_value
            else:
                T = self.thresholdRisk * incumbent_value

            if self.prints: print("EAR to " + str(targetStage) + " = " + str(EAR)  + " and  threshold=" + str(T)  )

            if EAR >= T:
                #not to jump to this target stage -> return the previous stage
                if targetStage-1>self.stage and np.random.uniform(0, 1) < 1-self.random_fraction_to_not_jump: 
                    # if jump -> process results
                    if self.prints: print("Jump to " + str(targetStage-1) + " (with EAR of " + str(EAR) + ")" )

                    self.process_results_to_jump(targetStage-1, prevSEL, prevUNSEL_cumulative)
                    SEL, UNSEL = prevSEL, prevUNSEL
                break 

            else:
                if targetStage == len(self.num_configs)-1: # last stage
                    prevUNSEL_cumulative += UNSEL
                
                    if incumbent_conf is not None: 
                        sel = [[incumbent_conf, incumbent_value, None]]
                        unsel = SEL #[[conf[0], conf[1], conf[2]] for conf in SEL]


                        EAR += self.risk(sel, unsel, targetStage, direction, study, trial)[2] 
                        
                        if direction == 'MAXIMIZE':
                            if incumbent_value==0: incumbent_value=0.01
                            T = self.thresholdRisk / incumbent_value
                        else:
                            T = self.thresholdRisk * incumbent_value

                        if EAR<T and np.random.uniform(0, 1) < 1-self.random_fraction_to_not_jump: 
                        #jump to next bracket -> SEL = []  UNSEL = all_configs
                            prevUNSEL_cumulative += SEL
                            if self.prints: print("Jump all to next bracket- (with EAR of " + str(EAR) + ")" )
                            #self.process_results_to_jump(targetStage, SEL, prevUNSEL_cumulative)
                            SEL = []

                    else:
                        if self.prints: print("Jump to last stage (with EAR of " + str(EAR) + ")" )
                    self.process_results_to_jump(targetStage, SEL, prevUNSEL_cumulative)

                    break

                else:
                    #continue to the next stage/budget
                    prevSEL = copy.deepcopy(SEL)
                    prevUNSEL = copy.deepcopy(UNSEL)
                    prevUNSEL_cumulative += UNSEL

                    _allConfigs  = [conf for conf, _, _ in SEL]
                    currentBudget = self.budgets[targetStage]
                    _configs, _losses, _stds  = self.predict(_allConfigs, study, trial, currentBudget)

                    #all_configs = [[_configs,_losses, _std] for _configs,_losses,_std in aux_allConfigs]
                    all_configs = []
                    for i in range(0, len(_configs)):
                        all_configs.append([_configs[i], float(_losses[i]), float(_stds[i])])

        return SEL, UNSEL


    def runNextConfig(self, study: Study, trial: FrozenTrial, SEL: list, UNSEL: list):
        ############################################
        # sort configs to test
        #############################################
        direction = "MAXIMIZE" if "MAXIMIZE" in str(study.direction) else 'MINIMIZE'
        if self.prints: print('Selecting the next configuration to test...  no queue configs ' + str(len(self.configQueued)))

        if len(self.configQueued)<=0: #no configs queued
            return None 

        if self.num_configs[self.stage] == self.done_num_configs[self.stage]: # all configs in tehe stage were tun
            return None

        if len(self.configQueued)==1: # just one config to run
            self.num_running += 1
            job = self.configQueued.pop(0) #(toAdd, budget) 
            if job[0] == {}: return {}

            job[0][self.budgetName] = job[1]
            return job[0]

        #here, there is a more than 2 configs to run
        if not self.modelIsTrained and len(self.configRunAll) > self.min_number_sample_model:
            self.trainModel(study) # train the model
        #else:
        if not self.modelIsTrained:
            self.num_running += 1
            job = self.configQueued.pop(0) #(toAdd, budget) 
            if job[0] == {}: return {}

            job[0][self.budgetName] = job[1]
            return job[0]
            
        def sortFirst_(val): 
            return val[1] 

        #run the config that is predicted the yield an higher risk reduction 
        untested_configs, untested_values, untested_std = self.get_untested_configs(study, trial)

        #if in the last stage
        if self.stage == len(self.num_configs)-1:
            #run the config with higher improvement
            listConf = []

            #untested configs in unsel
            aux_unsel = []
            for i in range(0, len(untested_configs)):
                aux_unsel.append([untested_configs[i], untested_values[i], untested_std[i]])

            #last stage so all unsel are untested, and sel is inc
            incumbent_conf, incumbent_value = self.get_incumbent(direction) 
            if incumbent_value is not None: 
                sel = [[incumbent_conf, incumbent_value, None]]
                process_list = []
                if __name__ == 'optuna.samplers._hyperjump._hyperjump':
                    for i in range(len(aux_unsel)): # only untested configs in unsel                                    
                        unsel = copy.deepcopy(aux_unsel)
                        unsel[i][2] = None

                        if self.pool is None:   
                            result = self.risk(sel, unsel, self.stage, self.budgets, study, trial)
                        else:
                            result = self.pool.apply_async(self.risk, (sel, unsel, self.stage, self.budgets, study, trial)) 
                        process_list.append(result)

                    for i in range(len(process_list)):
                        if self.pool is None:   
                            res = process_list[i]
                        else:
                            res = process_list[i].get()
                        if isinstance(res, list):
                            listConf.append([res[1][i][0], res[2]])

            else:
                for i in range(len(aux_unsel)):
                    listConf.append([aux_unsel[i][0], aux_unsel[i][1]]) #id, loss


            if len(listConf) != 0:
                listConf.sort(key = sortFirst_) 
                config = listConf[0][0]
            else:
                #sanity check
                aux_unsel.sort(key=sortFirst_) 
                config = aux_unsel[0][0]

            self._remove(config)
            return config



        ## not in last stage
        if not SEL or not UNSEL :
            tested_configs, tested_values = self.get_tested_configs()
            all_configs = []
            for i in range(0, len(tested_configs)):
                all_configs.append([tested_configs[i], tested_values[i], None])

            for i in range(0, len(untested_configs)):
                all_configs.append([untested_configs[i], untested_values[i], untested_std[i]])

            m = self.num_configs[self.stage+1] #no of configs to test in next stage
            aux_sel, aux_unsel = self.createSEL(all_configs, m, direction, study, trial)

        else:
            aux_sel = SEL
            aux_unsel = UNSEL

            for i in range(len(aux_sel)):
                _, c, s = self.predict([aux_sel[i][0]], study, trial, self.budgets[self.stage])
                aux_sel[i][1] = float(c[0])
                if aux_sel[i][2] is not None: aux_sel[i][2] = float(s[0])
                    
            for i in range(len(aux_unsel)):
                _, c, s = self.predict([aux_unsel[i][0]], study, trial, self.budgets[self.stage])
                aux_unsel[i][1] = float(c[0])
                if aux_unsel[i][2] is not None: aux_unsel[i][2] = float(s[0])

        # we have now sel and unsel 
        listConf = []
        process_list = []
        if __name__ == 'optuna.samplers._hyperjump._hyperjump':
            #SIMULATE THE RISK - set the models uncertainty of untested confgis to None
            l_aux = []
            for i in range(len(aux_sel)):
                if aux_sel[i][2] is None: continue # tested
                
                sel = copy.deepcopy(aux_sel)
                sel[i][2] = None
                
                if self.pool is None:   
                    result = self.risk(sel, aux_unsel, self.stage, direction, study, trial)
                else:
                    result = self.pool.apply_async(self.risk, (sel, aux_unsel, self.stage, direction, study, trial)) 
                process_list.append(result)
                l_aux.append(i)

            for j in range(len(process_list)):
                if self.pool is None:
                    res = process_list[j]
                else:
                    res = process_list[j].get()
                if isinstance(res, list):
                    listConf.append([res[0][l_aux[j]][0], res[2]])

            l_aux.clear()
            process_list.clear()
            for i in range(len(aux_unsel)):
                if aux_unsel[i][2] is None: continue # tested

                unsel = copy.deepcopy(aux_unsel)
                unsel[i][2] = None

                if self.pool is None:
                    result = self.risk(aux_sel, unsel, self.stage, direction, study, trial)
                else:
                    result = self.pool.apply_async(self.risk, (aux_sel, unsel, self.stage, direction, study, trial)) 
                process_list.append(result)
                l_aux.append(i)

            for j in range(len(process_list)):
                if self.pool is None:
                    res = process_list[j]
                else:
                    res = process_list[j].get()
                if isinstance(res, list):
                    listConf.append([res[1][l_aux[j]][0], res[2]])


            if len(listConf) > 0:
                listConf.sort(key=sortFirst_) 
            else:
                listConf = []
                for i in range(0, len(untested_configs)):
                    listConf.append([untested_configs[i], untested_values[i]])

                listConf.sort(key=sortFirst_) 
            

            if len(listConf) == 0: return None
            config = listConf[0][0]
            self._remove(config)
            return config

        return None


    def get_untested_configs(self, study: Study, trial: FrozenTrial):
        budget = self.budgets[self.stage]
        #print(budget)
        #print(self.configQueued )
        configs = [config for config, bud in self.configQueued if bud==budget]
        #print("get_untested_configs " + str(configs))

        if len(configs)==0: return [],[],[]

        configs2pred = np.array([]) if self.model2Use == 'EasyGPModel' else torch.tensor([])
        if trial.distributions == {}:
            #we need this because in the first trial, this parameter is {}
            all_trials = study.trials
            if len(all_trials) == 0:  return [],[],[]
            trial = all_trials[0]

        for conf in configs:
            sorted_keys = sorted(conf.keys())
            conf_sorted = {key: conf[key] for key in sorted_keys}

            if self.model2Use == 'EasyGPModel':
                sample = np.array([self.config_to_array(conf_sorted, study, trial)])
                configs2pred = np.concatenate((configs2pred, sample), axis=0) if len(configs2pred)>0 else sample
            else:
                sample = torch.tensor([self.config_to_array(conf_sorted, study, trial)])
                configs2pred = torch.cat((configs2pred, sample), dim=0)
        values, stds = self.model.eval_gp(configs2pred)
        return copy.deepcopy(configs), values, stds


    def get_tested_configs(self):
        budget = self.budgets[self.stage]
        configs = [config['config'] for config in self.configRun if config['config'][self.budgetName]==budget]
        values = np.array([config['value'] for config in self.configRun if config['config'][self.budgetName]==budget])

        return copy.deepcopy(configs), copy.deepcopy(values)
    
    
    def get_incumbent(self, direction):   
        """
            get the incumbent on the full budget
        """    
        budget = self.budgets[-1] # incumbent is on full budget

        incumbent_value = None
        incumbent_conf = None
        for config in self.configRunAll:
            if config['config'][self.budgetName]!=budget: continue

            value = config['value']
            if incumbent_value is None: 
                incumbent_value=value
                incumbent_conf = config['config']

            if direction == 'MAXIMIZE':
                if value > incumbent_value: 
                    incumbent_value=value
                    incumbent_conf = config['config']
            else:
                if value < incumbent_value: 
                    incumbent_value=value
                    incumbent_conf = config['config']

        return incumbent_conf, incumbent_value
    

    def process_results_to_jump(self, targetStage, SEL, UNSEL):
        """
            function that is called when it is predicted that we should jump for a taget budget

            Terminate configs that were queded (let run until the end the ones that are not finished)
            SEL configs to continue to the next stage
        """
        prints = False

        # terminate all configs in lower budgets than targetStage
        for stage in range(len(self.budgets)):
            if stage < targetStage:
                self.actual_num_configs[stage] = self.num_configs[stage]
        if targetStage==len(self.budgets)-1 and len(SEL)==0:
            # jump to next bracket
            self.actual_num_configs[targetStage] = self.num_configs[targetStage]


        if prints:print('Process results to jump')
        #remove the configs in queue
        budget = self.budgets[self.stage]
        for config, _, _ in UNSEL:
            config[self.budgetName] = budget
            self._remove(config)

        for config, _, _ in SEL:
            self._remove(config)
            config[self.budgetName] = budget
            self._remove(config)

        self.stage = targetStage # next stage jump
        budget = self.budgets[self.stage]
        for config, _, _ in SEL:
            self.actual_num_configs[self.stage] += 1
            config[self.budgetName] = budget
            self._queue(config, budget)     


    def risk(self, SEL, UNSEL, stage, direction, study: Study, trial: FrozenTrial):
        if not self.modelIsTrained and len(self.configRunAll) > self.min_number_sample_model:
            self.trainModel(study) # train the model

        SEL_BudgetMax = copy.deepcopy(SEL)
        Max_tested_SEL = -1 #if direction == 'MAXIMIZE' else 10000

        no_tested_SEL = 0
        no_untested_SEL = 0

        for i, (config, value, std) in enumerate(SEL_BudgetMax):

            if std is None: #tested config
                no_tested_SEL += 1

                if direction == 'MAXIMIZE':
                    SEL_BudgetMax[i][1] = value
                    if value >  Max_tested_SEL:
                        Max_tested_SEL = value
                else:
                    SEL_BudgetMax[i][1] = -value
                    if -value >  Max_tested_SEL:
                        Max_tested_SEL = -value

                SEL_BudgetMax[i][2] = 0.0 #std

            else:
                no_untested_SEL += 1
                prediction_ = self.predict([config], study, trial, self.budgets[stage])
                value, std = float(prediction_[1][0]), float(prediction_[2][0]) #mean and std

                SEL_BudgetMax[i][1] = value if direction == 'MAXIMIZE' else -value
                SEL_BudgetMax[i][2] = std if std >= 0.05 else  0.05
                   

        UNSEL_BudgetMax = copy.deepcopy(UNSEL)
        Max_tested_UNSEL = -1
        no_tested_UNSEL = 0
        no_untested_UNSEL = 0


        for i, (config, value, std) in enumerate(UNSEL_BudgetMax):

            if std is None: #tested config
                no_tested_UNSEL += 1

                if direction == 'MAXIMIZE':
                    UNSEL_BudgetMax[i][1] = value
                    if value >  Max_tested_UNSEL:
                        Max_tested_UNSEL = value
                else:
                    UNSEL_BudgetMax[i][1] = -value
                    if -value >  Max_tested_UNSEL:
                        Max_tested_UNSEL = -value
                
                UNSEL_BudgetMax[i][2] = 0.0
            
            else:
                no_untested_UNSEL += 1
                prediction_ = self.predict([config], study, trial, self.budgets[stage])
                value, std = float(prediction_[1][0]), float(prediction_[2][0]) #mean and std

                UNSEL_BudgetMax[i][1] = value if direction == 'MAXIMIZE' else -value
                UNSEL_BudgetMax[i][2] = std if std >= 0.05 else  0.05

        #return SEL_BudgetMax, UNSEL_BudgetMax, no_untested_SEL, no_untested_UNSEL, Max_tested_SEL, Max_tested_UNSEL
        area = risk_computation(SEL_BudgetMax, UNSEL_BudgetMax, no_untested_SEL, no_untested_UNSEL, Max_tested_SEL, Max_tested_UNSEL)
        return [SEL, UNSEL, area]


    def createSEL(self, all_configs, noConfSel, direction, study: Study, trial: FrozenTrial):
        ################################################
        #       order configs to sel
        ################################################
        def sortFirst(val): 
            return val[1] 

        def sortSecond(val): 
            return val[2]

        def sortThird(val): 
            return val[3] 

        eaRMatrix = []
        process_list = []

        if direction == 'MAXIMIZE':
            all_configs.sort(key = sortFirst, reverse=True) #sort configs by value and maximize
        else:
            all_configs.sort(key = sortFirst) #sort configs by loss and minimize


        sel = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
        unsel = all_configs[noConfSel:] #predicted configs to be dropped in the current stage
        if __name__ == 'optuna.samplers._hyperjump._hyperjump':
            if self.pool is None:
                result = self.risk(sel, unsel, self.stage, direction, study, trial)
            else:
                result = self.pool.apply_async(self.risk, (sel, unsel, self.stage, direction, study, trial)) 
            process_list.append(result)

            no_stages = len(self.num_configs)
            for i in range(1, no_stages):
                k = int(noConfSel / (self.eta**i))
                if k < 1: break
                
                sel_val_per = copy.deepcopy(sel[:-k])
                sel_val_rem = copy.deepcopy(sel[-k:])

                unsel_val_rem = copy.deepcopy(unsel[:k])
                unsel_val_per = copy.deepcopy(unsel[k:])

                sel_val = sel_val_per + unsel_val_rem 
                unsel_val = sel_val_rem + unsel_val_per

                if self.pool is None:
                    result = self.risk, (sel_val, unsel_val, self.stage, direction, study, trial)
                else:
                    result = self.pool.apply_async(self.risk, (sel_val, unsel_val, self.stage, direction, study, trial)) 
                process_list.append(result)
                
                if direction == 'MAXIMIZE':
                ## WE ARE USING VALUES and MAXIMIZE

                    sel_ucb = copy.deepcopy(sel)
                    unsel_lcb = copy.deepcopy(unsel)

                    for j in range(len(sel_ucb)):
                        std = 0.0 if sel_ucb[j][2] is None else sel_ucb[j][2]
                        sel_ucb[j].append(sel_ucb[j][1] + 1.645*std)

                    for j in range(len(unsel_lcb)):
                        std = 0.0 if unsel_lcb[j][2] is None else unsel_lcb[j][2]
                        unsel_lcb[j].append(unsel_lcb[j][1] - 1.645*std)


                    sel_ucb.sort(key = sortThird) #sort configs by lcb
                    unsel_lcb.sort(key = sortThird) #sort configs by ucb

                    sel_ucb_per = copy.deepcopy(sel_ucb[:-k])
                    sel_ucb_rem = copy.deepcopy(sel_ucb[-k:])

                    unsel_lcb_rem = copy.deepcopy(unsel_lcb[:k])
                    unsel_lcb_per = copy.deepcopy(unsel_lcb[k:])

                    sel_ucb_ = [] 
                    for j in range(len(sel_ucb_per)):
                        sel_ucb_.append(sel_ucb_per[j][0:3])
                    for j in range(len(unsel_lcb_rem)):
                        sel_ucb_.append(unsel_lcb_rem[j][0:3])

                    unsel_lcb_ = [] 
                    for j in range(len(unsel_lcb_per)):
                        unsel_lcb_.append(unsel_lcb_per[j][0:3])
                    for j in range(len(sel_ucb_rem)):
                        unsel_lcb_.append(sel_ucb_rem[j][0:3])

                    if self.pool is None:
                        result = self.risk, (sel_ucb_, unsel_lcb_, self.stage, direction, study, trial)
                    else:
                        result = self.pool.apply_async(self.risk, (sel_ucb_, unsel_lcb_, self.stage, direction, study, trial)) 
                    process_list.append(result)

                else:
                ## WE ARE USING LOSS and MINIMIZE
                    sel_lcb = copy.deepcopy(sel)
                    unsel_ucb = copy.deepcopy(unsel)

                    for j in range(len(sel_lcb)):
                        std = 0.0 if sel_lcb[j][2] is None else sel_lcb[j][2]
                        sel_lcb[j].append(sel_lcb[j][1] - 1.645*std)

                    for j in range(len(unsel_ucb)):
                        std = 0.0 if unsel_ucb[j][2] is None else unsel_ucb[j][2]
                        unsel_ucb[j].append(unsel_ucb[j][1] + 1.645*std)

                    sel_lcb.sort(key = sortThird) #sort configs by ucb
                    unsel_ucb.sort(key = sortThird) #sort configs by lcb

                    sel_lcb_per = copy.deepcopy(sel_lcb[:-k])
                    sel_lcb_rem = copy.deepcopy(sel_lcb[-k:])

                    unsel_ucb_rem = copy.deepcopy(unsel_ucb[:k])
                    unsel_ucb_per = copy.deepcopy(unsel_ucb[k:])

                    sel_lcb_ = [] 
                    for j in range(len(sel_lcb_per)):
                        sel_lcb_.append(sel_lcb_per[j][0:3])
                    for j in range(len(unsel_ucb_rem)):
                        sel_lcb_.append(unsel_ucb_rem[j][0:3])


                    unsel_ucb_ = [] 
                    for j in range(len(unsel_ucb_per)):
                        unsel_ucb_.append(unsel_ucb_per[j][0:3])
                    for j in range(len(sel_lcb_rem)):
                        unsel_ucb_.append(sel_lcb_rem[j][0:3])

                    if self.pool is None:
                        result = self.risk(sel_lcb_, unsel_ucb_, self.stage, direction, study, trial)
                    else:
                        result = self.pool.apply_async(self.risk, (sel_lcb_, unsel_ucb_, self.stage, direction, study, trial)) 
                    process_list.append(result)

            for i in range(len(process_list)):
                if self.pool is None:
                    res = process_list[i]
                else:
                    res = process_list[i].get()
                if isinstance(res, list):
                    eaRMatrix.append([res[0], res[1], res[2]])

        if len(eaRMatrix) > 0:
            # EAR is a matrix and each column has a possible sel=eaRMatrix[i][0], unsel=eaRMatrix[i][1], risk=eaRMatrix[i][2]]
            eaRMatrix.sort(key=sortSecond)  

            SEL = eaRMatrix[0][0]  #predicted configs to transit to the next stage
            UNSEL = eaRMatrix[0][1] #predicted configs to be dropped in the current stage

        else:
            SEL = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
            UNSEL = all_configs[noConfSel:] #predicted configs to be dropped in the current stage


        return SEL, UNSEL


def risk_computation(SEL_BudgetMax, UNSEL_BudgetMax, no_untested_SEL, no_untested_UNSEL, Max_tested_SEL, Max_tested_UNSEL):
    if no_untested_SEL == 0:
        #only tested config in SEL

        if no_untested_UNSEL == 0 :
            #only tested config in UNSEL
            return 0
        
        else:
            #untested and tested configs in UNSEL
            #sel is a dirac

            lib = ctypes.CDLL(os.path.abspath(path_to_risk_c + '/func.so'))
            #lib = ctypes.CDLL(os.path.abspath('func.so'))
            lib.fd.restype = ctypes.c_double
            lib.fd.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

            #array
            size = 3 + 1 + 2*len(UNSEL_BudgetMax)
            data = (ctypes.c_double * size)()
            #DATA  array 
            # [total configs,  No SEL,   No UNSEL,  SEL   ,   UNSEL]
            # [total configs,  0,   No UNSEL,  SEL   ,   UNSEL]
            # [total configs, No SEL, No UNSEL, SEL_max, UNSEL[0][0], UNSEL[0][1], UNSEL[1][0], UNSEL[1][1]]

            data[0] = size
            data[1] = 0
            data[2] = len(UNSEL_BudgetMax)
            data[3] = Max_tested_SEL

            count = 4
            for i in range(len(UNSEL_BudgetMax)):
                data[count] = UNSEL_BudgetMax[i][1]
                data[count + 1] = UNSEL_BudgetMax[i][2]
                count += 2


            try:
                data_= ctypes.cast(data, ctypes.c_void_p)
                func = LowLevelCallable(lib.fd, data_ )
                area = quad(func, 0, mp.inf, limit=1000) #, epsabs=1e-12)

            except Exception:
                return 1


    elif no_untested_UNSEL == 0: 
        #only tested config in SEL

        if no_untested_SEL == 0 :
            #only tested config in UNSEL
            return 0
        else:
            #untested and tested configs in SEL
            #unsel is a dirac
            lib = ctypes.CDLL(os.path.abspath(path_to_risk_c + '/func.so'))
            #lib = ctypes.CDLL(os.path.abspath('func.so'))
            lib.fd.restype = ctypes.c_double
            lib.fd.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

            #array
            size = 3 + 2*len(SEL_BudgetMax) + 1
            data = (ctypes.c_double * size)()

            data[0] = size
            data[1] = len(SEL_BudgetMax)
            data[2] = 0

            count = 3
            for i in range(len(SEL_BudgetMax)):
                data[count] = SEL_BudgetMax[i][1]
                data[count + 1] = SEL_BudgetMax[i][2]
                count += 2
            
            data[-1] = Max_tested_UNSEL

            try:
                data_= ctypes.cast(data, ctypes.c_void_p)
                func = LowLevelCallable(lib.fd, data_ )
                area = quad(func, 0, mp.inf, limit=1000) #, epsabs=1e-12)

            except Exception:
                return 1


    else:
        #there tested and untested configs in both sets

        lib = ctypes.CDLL(os.path.abspath(path_to_risk_c + '/func.so'))
        #lib = ctypes.CDLL(os.path.abspath('func.so'))
        lib.f.restype = ctypes.c_double
        lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

        #array
        size = 3 + 2*len(SEL_BudgetMax) + 2*len(UNSEL_BudgetMax)
        data = (ctypes.c_double * size)()
        #DATA  array 
        # [No SEL, No UNSEL,  SEL   ,   UNSEL]
        # [No SEL, No UNSEL, SEL[0][0], SEL[0][1], SEL[1][0], SEL[1][1], UNSEL[0][0], UNSEL[0][1], UNSEL[1][0], UNSEL[1][1]]

        data[0] = size
        data[1] = len(SEL_BudgetMax)
        data[2] = len(UNSEL_BudgetMax)

        count = 3
        for i in range(len(SEL_BudgetMax)):
            data[count] = SEL_BudgetMax[i][1]
            data[count + 1] = SEL_BudgetMax[i][2]
            count += 2

        for i in range(len(UNSEL_BudgetMax)):
            data[count] = UNSEL_BudgetMax[i][1]
            data[count + 1] = UNSEL_BudgetMax[i][2]
            count += 2
        
        data_= ctypes.cast(data, ctypes.c_void_p)
        func = LowLevelCallable(lib.f, data_ )
        opts = {"limit":2500, "epsabs": 1e-6}
        #opts = {"limit":1000} #, "epsabs":1e-12}

        if Max_tested_UNSEL != -1 and Max_tested_SEL == -1:
        #sel has only untested
        #unsel has untested and tested
            try:
                area =  nquad(func, [[Max_tested_UNSEL, np.inf],[0, np.inf]], opts=opts)
            except Exception:
                return 1

        elif Max_tested_UNSEL == -1 and Max_tested_SEL != -1:
        #unsel has untested
        #sel has untested and tested   
            def bounds_k(x):
                return [x-Max_tested_SEL, np.inf]

            try:
                area =  nquad(func, [bounds_k ,[0, np.inf]], opts=opts)
            except Exception:
                return 1

        elif Max_tested_UNSEL != -1 and Max_tested_SEL != -1:
        #unsel has untested and tested
        #sel has untested and tested 
            def bounds_k1(x):
                if x-Max_tested_SEL >  Max_tested_UNSEL:
                    return [x-Max_tested_SEL, np.inf]
                else:
                    return [Max_tested_UNSEL, np.inf]
            
            try:
                area =  nquad(func, [bounds_k1 , [0, np.inf]], opts=opts)
            except Exception:
                return 1

        else:

            try:
                area =  nquad(func, [[-np.inf, np.inf],[0, np.inf]], opts=opts)        
            except Exception:
                return 1

    return area[0]