from copy import deepcopy

def create_SAC_Classifier_algorithm(variant, *args, **kwargs):
    from .sac_classifier import SACClassifier

    algorithm = SACClassifier(*args, **kwargs)

    return algorithm

def create_RAQ_algorithm(variant, *args, **kwargs):
    from .raq import RAQ

    algorithm = RAQ(*args, **kwargs)

    return algorithm

def create_VICE_algorithm(variant, *args, **kwargs):
    from .vice import VICE

    algorithm = VICE(*args, **kwargs)

    return algorithm

def create_VICE_GAN_algorithm(variant, *args, **kwargs):
    from .vice_gan import VICEGAN

    algorithm = VICEGAN(*args, **kwargs)

    return algorithm

def create_VICE_RAQ_algorithm(variant, *args, **kwargs):
    from .viceraq import VICERAQ

    algorithm = VICERAQ(*args, **kwargs)

    return algorithm

def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL

    algorithm = SQL(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'SQL': create_SQL_algorithm,
    'SACClassifier': create_SAC_Classifier_algorithm,
    'RAQ': create_RAQ_algorithm,
    'VICE': create_VICE_algorithm,
    'VICEGAN': create_VICE_GAN_algorithm,
    'VICERAQ': create_VICE_RAQ_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
