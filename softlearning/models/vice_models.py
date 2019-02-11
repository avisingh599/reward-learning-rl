from softlearning.models.feedforward import feedforward_model


def create_feedforward_reward_classifier(observation_shape,
                                  *args,
                                  observation_preprocessor=None,
                                  name='feedforward_classifier',
                                  **kwargs):
    input_shapes = (observation_shape, )
    preprocessors = (observation_preprocessor, None)
    return feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        **kwargs)
