from skdecide import rollout_episode
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP
from skdecide.hub.solver.do_solver.do_solver_scheduling import DOSolver, SolvingMethod
from skdecide.hub.solver.sgs_policies.sgs_policies import PolicyMethodParams, BasePolicyMethod
from skdecide.hub.solver.gphh.gphh_po import GPHH_Pareto, feature_average_resource_requirements, \
    feature_n_predecessors, feature_n_successors, feature_task_duration, \
    feature_total_n_res, FeatureEnum, ParametersGPHH, protected_div, max_operator, min_operator, PrimitiveSet, GPHHPolicy
import operator
import numpy as np
import json
import pickle
import os


def run_gphh():

    import time
    n_runs = 1
    makespans = []

    domain: RCPSP = load_domain("j601_1.sm")
    # domain: RCPSP = load_domain("j1201_9.sm")

    # training_domains_names = ["j601_"+str(i)+".sm" for i in range(2, 11)]
    training_domains_names = ["j601_2.sm"]

    training_domains = []
    for td in training_domains_names:
        training_domains.append(load_domain(td))

    for i in range(n_runs):

        domain.set_inplace_environment(False)
        state = domain.get_initial_state()

        with open('cp_reference_permutations') as json_file:
            cp_reference_permutations = json.load(json_file)

        # with open('cp_reference_makespans') as json_file:
        #     cp_reference_makespans = json.load(json_file)

        # start = time.time()

        solver = GPHH_Pareto(training_domains=training_domains,
                      domain_model=training_domains[0],
                      weight=-1,
                      verbose=True,
                      reference_permutations=cp_reference_permutations,
                      # reference_makespans=cp_reference_makespans,
                      training_domains_names=training_domains_names,
                      params_gphh=ParametersGPHH.fast_test()
                      # params_gphh=ParametersGPHH.default()
                      )


        solver.solve(domain_factory=lambda: domain)

        # end = time.time()

        # runtimes.append((end-start))

        # heuristic = solver.hof
        # print('ttype:', solver.best_heuristic)
        # file = open('./trained_gphh_heuristics/test_gphh_'+str(i)+'.pkl', 'wb')
        # # file = open('./test_gphh_heuristic_'+str(i)+'.pkl', 'wb')
        #
        # pickle.dump(dict(hof= heuristic), file)
        # file.close()

        solver.set_domain(domain)
        states, actions, values = rollout_episode(domain=domain,
                                                  max_steps=1000,
                                                  solver=solver,
                                                  from_memory=state,
                                                  verbose=True,
                                                  outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
        print("Cost :", sum([v.cost for v in values]))
        makespans.append(sum([v.cost for v in values]))

    print('makespans: ', makespans)



def run_features():
    domain: RCPSP = load_domain("j301_1.sm")
    task_id = 2
    total_nres = feature_total_n_res(domain, task_id)
    print('total_nres: ', total_nres)
    duration = feature_task_duration(domain, task_id)
    print('duration: ', duration)
    n_successors = feature_n_successors(domain, task_id)
    print('n_successors: ', n_successors)
    n_predecessors = feature_n_predecessors(domain, task_id)
    print('n_predecessors: ', n_predecessors)
    average_resource_requirements = feature_average_resource_requirements(domain, task_id)
    print('average_resource_requirements: ', average_resource_requirements)


if __name__ == "__main__":
    run_gphh()

