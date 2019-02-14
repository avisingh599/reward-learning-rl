from .simple_replay_pool import SimpleReplayPool


class ActiveReplayPool(SimpleReplayPool):
    def __init__(self, *args, **kwargs):
        super(ActiveReplayPool, self).__init__(*args, **kwargs)

        active_fields = {
            'is_goal': {
                'shape': (1, ),
                'dtype': 'bool'
            },
        }

        self.add_fields(active_fields)