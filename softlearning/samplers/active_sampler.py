from .simple_sampler import SimpleSampler


class ActiveSampler(SimpleSampler):
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
            'is_goal': [info['is_goal']],
            'infos': info,
        }

        return processed_observation