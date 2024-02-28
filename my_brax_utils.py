from typing import Optional, Dict, Any
import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

def get_inference_func(params_filepath: str, action_size: int, observation_size: int):
    def make_inference_fn(
    observation_size: int,
    action_size: int,
    normalize_observations: bool = True,
    network_factory_kwargs: Optional[Dict[str, Any]] = None,
    ):
        normalize = lambda x, y: x
        if normalize_observations:
            normalize = jax.nn.standardize
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=normalize,
            **(network_factory_kwargs or {}),
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)
        return make_policy

    config_dict = { #load_config_dict(checkpoint_path)
    'observation_size': observation_size,
    'action_size': action_size,
    'normalize_observations': False,
    'network_factory_kwargs': None,
    }

    make_policy = make_inference_fn(
        config_dict['observation_size'],
        config_dict['action_size'],
        config_dict['normalize_observations'],
        network_factory_kwargs=config_dict['network_factory_kwargs'], # {"policy_hidden_layer_sizes": (128,) * 4}
    )
    params = model.load_params(params_filepath)
    jit_inference_fn = jax.jit(make_policy(params, deterministic=True))
    return jit_inference_fn