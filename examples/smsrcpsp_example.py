from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_multiskill_domain
from skdecide.hub.domain.rcpsp.rcpsp_sk import build_stochastic_from_deterministic_ms, build_n_determinist_from_stochastic_ms, SMSRCPSP
from skdecide import rollout_episode
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


if __name__ == "__main__":
    smsrcpsp_rollout()
