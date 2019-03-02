from copy import deepcopy

from softlearning.preprocessors.utils import get_preprocessor_from_params

def build_metric_learner_from_variant(variant, env, evaluation_data):
    sampler_params = variant['sampler_params']
    metric_learner_params = variant['metric_learner_params']
    metric_learner_params.update({
        'observation_shape': env.observation_space.shape,
        'max_distance': sampler_params['kwargs']['max_path_length'],
        'evaluation_data': evaluation_data
    })

    metric_learner = MetricLearner(**metric_learner_params)
    return metric_learner


def get_model_from_variant(variant, env, *args, **kwargs):
    pass

def get_reward_classifier_from_variant(variant, env, *args, **kwargs):
    from .vice_models import create_feedforward_reward_classifier
    
    classifier_params = variant['classifier_params']
    classifier_type = classifier_params['type']
    classifier_kwargs = deepcopy(classifier_params['kwargs'])

    # TODO Avi maybe have some optional preprocessing
    preprocessor_params = classifier_kwargs.pop('preprocessor_params', None)
    preprocessor = get_preprocessor_from_params(env, preprocessor_params)

    return create_feedforward_reward_classifier(
        observation_shape=env.active_observation_shape,
        #action_shape=env.action_space.shape,
        *args,
        observation_preprocessor=preprocessor,
        **classifier_kwargs,
        **kwargs)
