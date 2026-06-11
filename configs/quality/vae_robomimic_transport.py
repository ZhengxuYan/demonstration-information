"""
Config for DemInf VAEs on bimanual RoboMimic Transport image datasets.

This uses shoulder camera views and the full 14D transport action.

Example:

python scripts/train.py \
  --config=configs/quality/vae_robomimic_transport.py:transport/mh,s,1,both,/path/to/robo_mimic_transport/1.0.0 \
  --path /iris/u/jasonyan/data/deminf_outputs/transport_mh_image_v15 \
  --name transport_mh_both_s_vae_seed1
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.robomimic_transport import robomimic_transport_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18, ResNet18Decoder
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec


DEFAULT_TRANSPORT_RLDS = "/iris/u/jasonyan/data/transport_mh_image_v15_rlds/robo_mimic_transport/1.0.0"


def _parse_dataset_specs(ds: str, dataset_path: str):
    specs = []
    for index, raw_spec in enumerate(dataset_path.split("::")):
        spec = raw_spec.strip()
        if not spec:
            continue
        if "=" in spec:
            name, path_and_weight = spec.split("=", 1)
        else:
            name = ds.replace("/", "_") if len(dataset_path.split("::")) == 1 else f"{ds.replace('/', '_')}_{index}"
            path_and_weight = spec

        path = path_and_weight
        weight = None
        if "@" in path_and_weight:
            path, weight_text = path_and_weight.rsplit("@", 1)
            weight = float(weight_text)

        cfg = dict(
            path=path,
            train_split="train",
            transform=ModuleSpec.create(robomimic_transport_dataset_transform),
        )
        if weight is not None:
            cfg["weight"] = weight
        specs.append((name.replace("/", "_"), cfg))

    if not specs:
        raise ValueError(f"No dataset specs parsed from {dataset_path}")
    return specs


def get_config(config_str="transport/mh,s,1,both"):
    parts = config_str.split(",")
    dataset_path = DEFAULT_TRANSPORT_RLDS
    if len(parts) == 2:
        ds, config_type = parts
        seed = 1
        camera = "both"
    elif len(parts) == 3:
        ds, config_type, seed = parts
        seed = int(seed)
        camera = "both"
    elif len(parts) == 4:
        ds, config_type, seed, camera = parts
        seed = int(seed)
    elif len(parts) == 5:
        ds, config_type, seed, camera, dataset_path = parts
        seed = int(seed)
    else:
        raise ValueError(
            "Expected config string env,type[,seed[,camera[,dataset_path]]], "
            f"for example transport/mh,s,1,both. Got: {config_str}"
        )
    assert config_type in {"i", "s", "a", "sa"}
    assert camera in {"wrist", "agent", "both"}

    cameras = ("agent", "wrist") if camera == "both" else (camera,)
    image_keys = [f"observation->image->{key}" for key in cameras]
    image_encoders = {key: ModuleSpec.create(ResNet18, num_kp=64) for key in image_keys}
    image_decoders = {key: ModuleSpec.create(ResNet18Decoder) for key in image_keys}
    image_weights = {key: 1 / 200 for key in image_keys}

    encoder_keys = {
        "i": image_encoders,
        "s": {"observation->state": None, **image_encoders},
        "a": {"action": None},
        "sa": {"observation->state": None, **image_encoders, "action": None},
    }[config_type]

    decoder_keys = {
        "i": image_decoders,
        "s": {"observation->state": None, **image_decoders},
        "a": {"action": None},
        "sa": {"observation->state": None, **image_decoders, "action": None},
    }[config_type]

    z_dim = {
        "i": 16,
        "s": 32,
        "a": 16,
        "sa": 48,
    }[config_type]

    weights = {
        "i": image_weights,
        "s": {"observation->state": 1.0, **image_weights},
        "a": {"action": 1.0},
        "sa": {"observation->state": 1.0, **image_weights, "action": 1.0},
    }[config_type]

    observation_structure = {
        "image": {key: (84, 84) for key in cameras},
    }
    if config_type in {"s", "sa"}:
        observation_structure = {
            "state": {
                StateEncoding.MISC: NormalizationType.GAUSSIAN,
            },
            **observation_structure,
        }

    structure = {
        "observation": observation_structure,
        "action": {
            "desired_delta": {
                StateEncoding.MISC: NormalizationType.GAUSSIAN,
            },
        },
    }

    dataset_specs = _parse_dataset_specs(ds, dataset_path)
    datasets = {name: cfg for name, cfg in dataset_specs}

    dataloader = dict(
        datasets=datasets,
        n_obs=1,
        n_action=1,
        augment_kwargs=dict(scale_range=(0.85, 0.95), aspect_ratio_range=None),
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,
        cache=True,
        prefetch=tf.data.AUTOTUNE,
    )

    alg = ModuleSpec.create(
        BetaVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders=encoder_keys,
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [512, 512], activate_final=True), flatten_time=True
            ),
        ),
        decoder=ModuleSpec.create(
            MultiDecoder,
            trunk=ModuleSpec.create(MLP, [512, 512], activate_final=True),
            decoders=decoder_keys,
        ),
        z_dim=z_dim,
        beta=0.01,
        weights=weights,
    )

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=ModuleSpec.create(optax.adam),
            lr_schedule=ModuleSpec.create(optax.constant_schedule, 0.0001),
            steps=100000,
            log_freq=500,
            val_freq=2500,
            save_freq=10000,
            val_steps=25,
            seed=int(seed),
        )
    )
