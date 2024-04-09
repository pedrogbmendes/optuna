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

import copy 
import numpy as np
from scipy.stats import norm

import torch
from optuna.samplers._bohb.gps import ExactGPModel, EasyGPModel



DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class BOHB(BaseSampler):
    """Sampler using BOHB.

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
                    num_samples = 100,
                    seed: Optional[int] = None) -> None:
        
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
        #First, verify if there are queued configs
        if len(self.configQueued)>0:
            self.num_running += 1
            job = self.configQueued.pop(0) #(toAdd, budget) self._run()
            if job[0] == {}: return {}

            job[0][self.budgetName] = job[1]
            return job[0]  # return (config, budget)
        

        if (self.actual_num_configs[self.stage] < self.num_configs[self.stage]):
            # queue is empty, so add configs to the queue
            config =  self._sample_relative(study, trial, search_space,)
            self.actual_num_configs[self.stage] += 1
            budget = self.budgets[self.stage]
            self._queue(config, budget) 
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
            self.trainModel() # build the model

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


    def _done(self, trial: FrozenTrial, value: float):
        # Sort the keys alphabetically
        job = trial.params
        sorted_keys = sorted(job.keys())
        job_sorted = {key: job[key] for key in sorted_keys}

        self.configRun.append({'config': job_sorted, 'value':value})
        self.configRunAll.append({'config': job_sorted, 'value':value, 'trial':trial})

        arrayConfig = self.config_to_array(job_sorted, trial)
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
        # number of 'SH rungs'
        self.currentIteration += 1
        s = self.max_SH_iter - 1 - (self.currentIteration%self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)

        self.num_configs = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        self.actual_num_configs = [0]*len(self.num_configs)
        self.budgets = self.allBudgets[(-s-1):]
        self.num_running = 0
        self.stage = 0
        self.configRun = []


    def nextStage(self, direction):
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
        self._done(trial, values[0])
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
        print("Loading data...")
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


    def trainModel(self, ) -> None:
        print("Train the model...")
        # Sort the keys alphabetically

        if self.model2Use == 'EasyGPModel':
            if self.model is None: self.model = EasyGPModel(normalize=True)

            if self.x is None:
                self.x =  np.array([self.config_to_array(config['config'], config['trial']) for config in self.configRunAll])
            if self.y is None:
                self.y = np.array([config['value'] for config in self.configRunAll])

        else:

            if self.x is None:
                self.x = torch.tensor([self.config_to_array(config['config'], config['trial']) for config in self.configRunAll])
            if self.y is None:
                self.y = torch.tensor([config['value'] for config in self.configRunAll])

            # initialize likelihood and model
            #if self.model is None:
            self.model = ExactGPModel(self.x, self.y) # create and train
            #self.model.train_gp() # just train

        self.model.train_gp(self.x, self.y)
        
        #print(self.model.eval_gp(self.x[0:1]))
        self.modelIsTrained = True


    def BO_EI(self, study: Study, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:

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
                    _sample = np.array([self.config_to_array(sample, trial_max)])
                    configs_ei = np.concatenate((configs_ei, _sample), axis=0) if len(configs_ei)>0 else _sample
                else:
                    _sample = torch.tensor([self.config_to_array(sample, trial_max)])
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
    
    
    def config_to_array(self, config, trial: FrozenTrial):
        # convert a config dict into a numpy array to be used in the models
        _list_config = []

        for hp_name, val in config.items():
            if isinstance(val, str): # not numeric
                choices = list(trial.distributions[hp_name].choices)
                it = 0
                for hp_val in choices:
                    if hp_val==val:
                        _list_config.append(it)
                        break
                    it += 1
            else:
                _list_config.append(val)
        #print(_list_config)
        return _list_config


    def getBudgets(self, ):
        return self.allBudgets