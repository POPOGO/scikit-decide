
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set
import random

from skdecide import Domain, Space, TransitionValue, Distribution, TransitionOutcome, ImplicitSpace, DiscreteDistribution
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, DeterministicInitialized, \
    Actions, FullyObservable, Markovian, Goals, Simulation, UncertainTransitions
from skdecide.builders.scheduling.modes import SingleMode, MultiMode
from skdecide.builders.scheduling.resource_consumption import VariableResourceConsumption, ConstantResourceConsumption
from skdecide.builders.scheduling.precedence import WithPrecedence
from skdecide.builders.scheduling.preemptivity import WithPreemptivity, WithoutPreemptivity
from skdecide.builders.scheduling.resource_type import WithResourceTypes, WithResourceUnits, WithoutResourceUnit
from skdecide.builders.scheduling.resource_renewability import RenewableOnly, MixedRenewable
from skdecide.builders.scheduling.scheduling_domains_modelling import SchedulingActionEnum, State, \
    SchedulingAction, SchedulingEvent, SchedulingEventEnum
from skdecide.builders.scheduling.task_duration import SimulatedTaskDuration, DeterministicTaskDuration, \
    UncertainUnivariateTaskDuration
from skdecide.builders.scheduling.task_progress import CustomTaskProgress, DeterministicTaskProgress
from skdecide.builders.scheduling.skills import WithResourceSkills, WithoutResourceSkills
from skdecide.builders.scheduling.time_lag import WithTimeLag, WithoutTimeLag
from skdecide.builders.scheduling.time_windows import WithTimeWindow, WithoutTimeWindow
from skdecide.builders.scheduling.preallocations import WithPreallocations, WithoutPreallocations
from skdecide.builders.scheduling.conditional_tasks import WithConditionalTasks, WithoutConditionalTasks
from skdecide.builders.scheduling.resource_availability import UncertainResourceAvailabilityChanges, \
    WithoutResourceAvailabilityChange, DeterministicResourceAvailabilityChanges
from skdecide.builders.scheduling.resource_costs import WithResourceCosts, WithoutResourceCosts, WithModeCosts, WithoutModeCosts

from skdecide import Distribution
from skdecide.builders.scheduling.scheduling_domains import SingleModeRCPSP, SchedulingDomain, SchedulingObjectiveEnum, SingleModeRCPSP_Stochastic_Durations, SingleModeRCPSP_Stochastic_Durations_WithConditionalTasks, SingleModeRCPSP_Simulated_Stochastic_Durations_WithConditionalTasks, MultiModeRCPSPWithCost, DeterministicSchedulingDomain, UncertainSchedulingDomain
from skdecide.builders.scheduling.modes import SingleMode, MultiMode, ModeConsumption, ConstantModeConsumption
from skdecide import rollout, rollout_episode

import numpy as np
from scipy.stats import expon
from scipy.stats import planck
from typing import Optional
from skdecide import DiscreteDistribution


class MyExampleRCPSPDomain_fixed_initial_resource_levels(SingleModeRCPSP):

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2,3], 2:[4], 3:[5], 4:[5], 5:[]}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1,2,3,4,5])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({'r1': 0, 'r2': 0}),
            2: ConstantModeConsumption({'r1': 1, 'r2': 1}),
            3: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            4: ConstantModeConsumption({'r1': 2, 'r2': 1}),
            5: ConstantModeConsumption({'r1': 0, 'r2': 0})
        }

    def _get_resource_types_names(self) -> List[str]:
        return ['r1', 'r2']

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        all_durations = {1: 0, 2: 5, 3: 6, 4: 4, 5: 0}
        return all_durations[task]

    def _get_original_fixed_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {'r1': 4, 'r2': 3}
        return all_resource_quantities[resource]

    def _get_next_resource_change_time_distribution(self, res: str, currenttime: int, previousresourcehangetime: Optional[int]):
        dist_vals = []
        for t in range(currenttime, self.get_max_horizon()):
            time_without_changes = t - previousresourcehangetime
            # p = expon.cdf(time_without_changes, scale=3)
            p = planck.cdf(time_without_changes, lambda_=0.5)
            if p > 0.99:
                p = 1.
            dist_vals.append((t, p))
            if p == 1.:
                break
        return DiscreteDistribution(dist_vals)

    def _get_next_resource_change_delta_distribution(self, res: str, change_time: int, previous_resource_event:(int, int)):
        if previous_resource_event is None:
            dist_vals = [(-1, 0.5), (1, 0.5)]
        else:
            previous_resource_event_delta = previous_resource_event[1]
            if previous_resource_event_delta < 0:
                dist_vals = [(previous_resource_event_delta - 1, 0.2),
                             (previous_resource_event_delta + 1, 0.8)]
            else:
                dist_vals = [(previous_resource_event_delta - 1, 0.8),
                             (previous_resource_event_delta + 1, 0.2)]
        return DiscreteDistribution(dist_vals)

    def _get_quantity_resource(self, res: str, time: int, next_resource_event: (int, int)):
        next_resource_event_t = next_resource_event[0]
        next_resource_event_delta = next_resource_event[1]
        if next_resource_event_t <= time:
            # Event happening before time -> need to account for the change
            val = max(0, self._get_quantity_resource(res, next_resource_event_t-1) + next_resource_event_delta)
        else:
            val = self._get_quantity_resource(res, next_resource_event_t-1)

        return val




class D(UncertainSchedulingDomain,
                      SingleMode,
                      DeterministicTaskDuration,
                      DeterministicTaskProgress,
                      WithoutResourceUnit,
                      WithoutPreallocations,
                      WithoutTimeLag,
                      WithoutTimeWindow,
                      WithoutResourceSkills,
                      # WithoutResourceAvailabilityChange,
                      WithoutConditionalTasks,
                      RenewableOnly,
                      ConstantResourceConsumption,  # problem with unimplemented classes with this
                      WithoutPreemptivity,  # problem with unimplemented classes with this
                      WithoutModeCosts,
                      WithoutResourceCosts
                      ):
    pass

class MyExampleRCPSPDomain_varying_initial_resource_levels(D):

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 500

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2,3], 2:[4], 3:[5], 4:[5], 5:[]}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1,2,3,4,5])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({'r1': 0, 'r2': 0}),
            2: ConstantModeConsumption({'r1': 1, 'r2': 1}),
            3: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            4: ConstantModeConsumption({'r1': 2, 'r2': 1}),
            5: ConstantModeConsumption({'r1': 0, 'r2': 0})
        }

    def _get_resource_types_names(self) -> List[str]:
        return ['r1', 'r2']

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        all_durations = {1: 0, 2: 50, 3: 60, 4: 40, 5: 0}
        return all_durations[task]

    def _get_original_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        all_resource_quantities = {'r1': [2 for i in range(self.get_max_horizon())],
                                   'r2': [2 for i in range(self.get_max_horizon())]}
        return all_resource_quantities[resource][time]

    def _get_next_resource_change_time_distribution(self, res: str, currenttime: int, previous_resource_event: Optional[SchedulingEvent] = None):
        if previous_resource_event is None:
            previousresourcehangetime = 0
        else:
            previousresourcehangetime = previous_resource_event.t
        dist_vals = []
        previous_val = 0.

        for t in range(currenttime+1, self.get_max_horizon()):
            time_without_changes = t - previousresourcehangetime
            # p = expon.cdf(time_without_changes, scale=3)
            p_cum = planck.cdf(k=time_without_changes, lambda_=0.05)
            p = p_cum - previous_val
            if p_cum > 0.99:
                p_cum = 1.
                p = 1.-previous_val
            previous_val = p_cum
            dist_vals.append((t, p))
            if p_cum == 1.:
                break
        print('next_changes: ', res, currenttime, previousresourcehangetime, dist_vals)
        assert sum([x[1] for x in dist_vals])
        return DiscreteDistribution(dist_vals)

    def _get_next_resource_change_delta_distribution(self, res: str, change_time: int, previous_resource_event: Optional[SchedulingEvent] = None):
        if previous_resource_event is None:
            dist_vals = [(-1, 0.5), (1, 0.5)]
        else:
            previous_resource_event_delta = previous_resource_event.resource_delta
            print('previous_resource_event: ', previous_resource_event_delta)
            if previous_resource_event_delta < 0:
                dist_vals = [(previous_resource_event_delta - 1, 0.2),
                             (previous_resource_event_delta + 1, 0.8)]
            else:
                dist_vals = [(previous_resource_event_delta - 1, 0.8),
                             (previous_resource_event_delta + 1, 0.2)]
        return DiscreteDistribution(dist_vals)

    def _get_quantity_resource(self, res: str, time: int, previous_resource_event: SchedulingEvent, next_resource_event: SchedulingEvent):
        if previous_resource_event is not None:
            previous_resource_event_t = previous_resource_event.t
            previous_resource_event_delta = previous_resource_event.resource_delta
        if next_resource_event is not None:
            next_resource_event_t = next_resource_event.t
            next_resource_event_delta = next_resource_event.resource_delta

        if next_resource_event is not None and next_resource_event_t <= time:
            val = max(0, self._get_original_quantity_resource(res, time) + next_resource_event_delta)
        elif previous_resource_event is not None and previous_resource_event_t <= time:
            val = max(0, self._get_original_quantity_resource(res, time) + previous_resource_event_delta)
        else:
            val = self._get_original_quantity_resource(res, time)

        return val


def run_example():

    domain = MyExampleRCPSPDomain_varying_initial_resource_levels()

    state = domain.get_initial_state()
    print("Initial state : ", state)
    # actions = domain.get_applicable_actions(state)
    # print([str(action) for action in actions.get_elements()])
    # action = actions.get_elements()[0]
    # new_state = domain.get_next_state(state, action)
    # print("New state ", new_state)
    # actions = domain.get_applicable_actions(new_state)
    # print("New actions : ", [str(action) for action in actions.get_elements()])
    # action = actions.get_elements()[0]
    # print(action)
    # new_state = domain.get_next_state(new_state, action)
    # print("New state :", new_state)
    # print('_is_terminal: ', domain._is_terminal(state))
    # ONLY KEEP LINE BELOW FOR SIMPLE ROLLOUT
    solver = None
    # UNCOMMENT BELOW TO USE ASTAR
    # domain.set_inplace_environment(False)
    # solver = lazy_astar.LazyAstar(from_state=state, heuristic=None, verbose=True)
    # solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(domain=domain,
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print(states[-1])


if __name__ == "__main__":
    run_example()
