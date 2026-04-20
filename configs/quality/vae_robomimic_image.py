"""
Config for training beta vae's on robomimic

Example command:

python scripts/train.py \
    --config=configs/quality/vae_robomimic_image.py:square/mh,s,1,wrist \
    --path test \
    --name state

Multiple RLDS roots can be mixed by separating dataset specs with "::":

python scripts/train.py \
    --config=configs/quality/vae_robomimic_image.py:combined,s,1,wrist,square_mh=/path/to/mh@72707::random=/path/to/random@8055 \
    --path test \
    --name combined_wrist

Use camera=both to train one fused observation VAE over both agent and wrist images.
Use type=i to train an image-only observation VAE without robot proprio.
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18, ResNet18Decoder
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec


DEFAULT_ROBOMIMIC_RLDS = "/iris/u/jasonyan/data/robomimic_rlds_v2/robo_mimic/1.0.0"


def _parse_dataset_specs(ds: str, dataset_path: str):
    """Parse one or more RLDS dataset roots.

    Format:
      /path/to/dataset
      name=/path/to/dataset
      name=/path/to/dataset@weight
      spec::spec::...

    Multi-dataset configs intentionally omit validation splits because custom
    RLDS builds often only contain train.
    """

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

        weight = None
        path = path_and_weight
        if "@" in path_and_weight:
            path, weight_text = path_and_weight.rsplit("@", 1)
            weight = float(weight_text)

        cfg = dict(
            path=path,
            train_split="train",
            transform=ModuleSpec.create(robomimic_dataset_transform),
        )
        if weight is not None:
            cfg["weight"] = weight
        specs.append((name.replace("/", "_"), cfg))

    if not specs:
        raise ValueError(f"No dataset specs parsed from {dataset_path}")
    return specs


def get_config(config_str="square/mh,sa,1"):
    parts = config_str.split(",")
    dataset_path = DEFAULT_ROBOMIMIC_RLDS
    if len(parts) == 2:
        ds, config_type = parts
        seed = 1
        camera = "wrist"
    elif len(parts) == 3:
        ds, config_type, seed = parts
        seed = int(seed)
        camera = "wrist"
    elif len(parts) == 4:
        ds, config_type, seed, camera = parts
        seed = int(seed)
    elif len(parts) == 5:
        ds, config_type, seed, camera, dataset_path = parts
        seed = int(seed)
    else:
        raise ValueError(
            "Expected config string env,type[,seed[,camera[,dataset_path]]], "
            f"for example square/mh,s,1,wrist. Got: {config_str}"
        )
    assert config_type in {"i", "s", "a", "sa"}
    assert camera in {"wrist", "agent", "both"}

    cameras = ("agent", "wrist") if camera == "both" else (camera,)
    image_keys = [f"observation->image->{key}" for key in cameras]
    dataset_specs = _parse_dataset_specs(ds, dataset_path)
    has_multiple_datasets = len(dataset_specs) > 1
    image_encoders = {key: ModuleSpec.create(ResNet18, num_kp=64) for key in image_keys}
    image_decoders = {key: ModuleSpec.create(ResNet18Decoder) for key in image_keys}
    image_weights = {key: 1 / 200 for key in image_keys}

    encoder_keys = {
        "i": image_encoders,
        "s": {"observation->state": None, **image_encoders},
        "a": {"action": None},
        "sa": {
            "observation->state": None,
            **image_encoders,
            "action": None,
        },
    }[config_type]

    decoder_keys = {
        "i": image_decoders,
        "s": {"observation->state": None, **image_decoders},
        "a": {"action": None},
        "sa": {
            "observation->state": None,
            **image_decoders,
            "action": None,
        },
    }[config_type]
    seed = int(seed)

    z_dim = {
        "i": 16,
        "s": 16,  # XYZ+ROT=6 + Object XYZ + Rot=6 + Gripper=1 -- Total 13
        "a": 6,  # XYZ+ROT=6
        "sa": 22,
    }[config_type]

    weights = {
        "i": image_weights,
        "s": {"observation->state": 1.0, **image_weights},
        "a": {"action": 1.0},
        "sa": {"observation->state": 1.0, **image_weights, "action": 1.0},
    }[config_type]

    # Define the structure
    observation_structure = {
        "image": {key: (84, 84) for key in cameras},
    }
    if config_type in {"s", "sa"}:
        observation_structure = {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_QUAT: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.GAUSSIAN,
            },
            **observation_structure,
        }

    structure = {
        "observation": observation_structure,
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.BOUNDS},
        },
    }

    datasets = {}
    for name, ds_cfg in dataset_specs:
        if not has_multiple_datasets:
            ds_cfg["val_split"] = "val"
        datasets[name] = ds_cfg

    dataloader = dict(
        datasets=datasets,
        n_obs=1,
        n_action=1,
        augment_kwargs=dict(scale_range=(0.85, 0.95), aspect_ratio_range=None),
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
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

    lr_schedule = ModuleSpec.create(optax.constant_schedule, 0.0001)
    optimizer = ModuleSpec.create(optax.adam)

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=100000,
            log_freq=500,
            val_freq=2500,
            save_freq=10000,
            val_steps=25,
            seed=seed,
        )
    )
