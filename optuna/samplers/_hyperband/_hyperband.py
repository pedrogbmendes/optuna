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

class HyperBand(BaseSampler):
    """Sampler using Hyerband.

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
                    seed: Optional[int] = None) -> None:
        
        self._rng = LazyRandomState(seed)
        #self._random_sampler = RandomSampler(seed=seed)
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

        #self.config_sampleIndependent = []


    def reseed_rng(self) -> None:
        self._rng.rng.seed()


    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        search_space: Dict[str, BaseDistribution] = {}
        a = self._search_space.calculate(study)
        for name, distribution in a.items():
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

        trans = _SearchSpaceTransform(search_space)
        trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
        return trans.untransform(trans_params)


    def _queue(self, toAdd, budget):
        self.configQueued.append((toAdd, budget))


    def _done(self, job, value):
        self.configRun.append({'config': job, 'value':value})
        self.configRunAll.append({'config': job, 'value':value})
        

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
        self._done(trial.params, values[0])

    def getBudgets(self, ):
        return self.allBudgets