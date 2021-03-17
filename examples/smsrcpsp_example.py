from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_multiskill_domain
from skdecide.hub.domain.rcpsp.rcpsp_sk import build_stochastic_from_deterministic_ms, build_n_determinist_from_stochastic_ms, SMSRCPSP
from skdecide import rollout_episode
from skdecide.hub.solver.gphh.gphh_po import ParametersGPHH, GPHH
import random


def smsrcpsp_rollout():
    file_name = '100_5_22_15.def'
    original_domain = load_multiskill_domain(file_name=file_name)
    print('original_domain: ', original_domain)
    task_to_noise = set(random.sample(original_domain.get_tasks_ids(), len(original_domain.get_tasks_ids())))
    # random.seed(1)
    stochastic_domain = build_stochastic_from_deterministic_ms(original_domain,
                                                                task_to_noise=task_to_noise)
    stochastic_domain.set_inplace_environment(True)
    deterministic_domains = build_n_determinist_from_stochastic_ms(stochastic_domain,
                                                                    nb_instance=5)

    print('Ouputing durations for 5 sampled domains:')
    for det_dom in deterministic_domains:
        print(det_dom)
        durations = []
        for t in det_dom.get_tasks_ids():
            durations.append(det_dom.get_task_duration(t))
        print(durations)

    stochastic_domain.set_inplace_environment(False)
    state = stochastic_domain.get_initial_state()
    solver = None
    states, actions, values = rollout_episode(domain=stochastic_domain,
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              verbose=False,
                                              outcome_formatter=lambda
                                                  o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("One random Walk complete")
    print("Cost :", sum([v.cost for v in values]))




def smsrcpsp_gphh():
    file_name = '100_5_22_15.def'
    original_domain = load_multiskill_domain(file_name=file_name)
    print('original_domain: ', original_domain)
    task_to_noise = set(random.sample(original_domain.get_tasks_ids(), len(original_domain.get_tasks_ids())))
    # random.seed(1)
    stochastic_domain = build_stochastic_from_deterministic_ms(original_domain,
                                                                task_to_noise=task_to_noise)
    stochastic_domain.set_inplace_environment(False)
    training_domains = build_n_determinist_from_stochastic_ms(stochastic_domain,
                                                                    nb_instance=5)
    training_domains = [original_domain]
    training_domains_names = [str(i) for i in range(len(training_domains))]


    ########### TODO: GPHH for SMSRCPSP
    solver = GPHH(
        training_domains=training_domains,
                  domain_model=original_domain,
                  weight=-1,
                  verbose=True,
                  reference_permutations={},
                  training_domains_names=training_domains_names,
                  params_gphh=ParametersGPHH.ms_default()
                  )

    solver.solve(domain_factory=lambda: original_domain)
    # solver = None # REMOVEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    ###########

    stochastic_domain.set_inplace_environment(False)
    state = stochastic_domain.get_initial_state()
    states, actions, values = rollout_episode(
        domain=original_domain,
        # domain=stochastic_domain,
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              verbose=False,
                                              outcome_formatter=lambda
                                                  o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("One gphh complete")
    print("Cost :", sum([v.cost for v in values]))
    print("max horizon: ", stochastic_domain.get_max_horizon())
    print("completed: ", len(states[-1].tasks_complete))
    from tests.test_scheduling import check_rollout_consistency
    check_rollout_consistency(stochastic_domain, states)

    all_durations = []
    for id in stochastic_domain.get_tasks_ids():
        if id in states[-1].tasks_complete:
            all_durations.append(states[-1].tasks_details[id].sampled_duration)
    print('all_durations: ', all_durations)
    print('sum_durations: ', sum(all_durations))


if __name__ == "__main__":
    # smsrcpsp_rollout()
    smsrcpsp_gphh()
