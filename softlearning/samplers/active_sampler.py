from .simple_sampler import SimpleSampler


class ActiveSampler(SimpleSampler):
    def __init__(self, is_goal_key='is_goal', **kwargs):
        super(ActiveSampler, self).__init__(**kwargs)
        self._is_goal_key = is_goal_key

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              terminal,
                              next_observation,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'is_goal': [info[self._is_goal_key]],
            'infos': info,
        }

        return processed_observation