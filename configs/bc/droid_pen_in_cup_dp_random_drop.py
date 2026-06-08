import os

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.oxe import droid_dataset_transform
from openx.data.filters import filter_by_scores
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads.ddpm import DDPMActionHead
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet34
from openx.networks.components.unet import ConditionalUnet1D
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec


DEFAULT_DATASET_PATH = "/iris/u/jasonyan/data/droid_pen_in_cup_06072026_rlds/droid_pen_in_cup/1.0.0"
DEFAULT_SCORE_ROOT = "/iris/u/jasonyan/data/droid_pen_in_cup_06072026_random_drop_scores"


def get_config(config_str="pen_in_cup,25,random,1"):
    parts = config_str.split(",")
    if len(parts) not in (4, 5, 6):
        raise ValueError(
            "Expected config_str env,drop_percent,estimator,seed[,score_root[,dataset_path]], "
            f"got {config_str!r}"
        )
    env, drop_percent, estimator, seed = parts[:4]
    score_root = parts[4] if len(parts) >= 5 else os.environ.get("PEN_IN_CUP_SCORE_ROOT", DEFAULT_SCORE_ROOT)
    dataset_path = parts[5] if len(parts) >= 6 else os.environ.get("PEN_IN_CUP_RLDS_PATH", DEFAULT_DATASET_PATH)
    seed = int(seed)

    drop_percent = float(drop_percent)
    if estimator == "random":
        filter_path = os.path.join(
            score_root,
            env.replace("/", "_"),
            estimator,
            "seed-" + str(seed),
            f"random_drop_{int(drop_percent):02d}_seed{seed}.pkl",
        )
    elif estimator in ("observability", "optimality"):
        filter_path = os.path.join(
            score_root,
            env.replace("/", "_"),
            estimator,
            f"{estimator}_scores.pkl",
        )
    else:
        raise ValueError(f"Unsupported estimator={estimator!r}; expected random, observability, or optimality")

    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.GAUSSIAN,
            },
            "image": {
                "agent_1": (128, 128),
                "wrist": (128, 128),
            },
        },
        "action": {
            "desired_absolute": {
                StateEncoding.EE_POS: NormalizationType.BOUNDS_5STDV,
                StateEncoding.EE_ROT6D: NormalizationType.BOUNDS_5STDV,
                StateEncoding.GRIPPER: NormalizationType.BOUNDS,
            },
        },
    }
    dataset_config = dict(
        path=dataset_path,
        train_split="train",
        transform=ModuleSpec.create(droid_dataset_transform),
    )
    if drop_percent > 0:
        dataset_config["train_filter"] = ModuleSpec.create(
            filter_by_scores, filter_path, "ep_idx", percentile=drop_percent
        )

    dataloader = dict(
        datasets={
            env.replace("/", "_"): dataset_config,
        },
        n_obs=2,
        n_action=16,
        augment_kwargs=dict(scale_range=(0.875, 0.975), aspect_ratio_range=(1.33, 1.333)),
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,
        cache=True,
        prefetch=tf.data.AUTOTUNE,
    )

    alg = ModuleSpec.create(
        BehaviorCloning,
        observation_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent_1": ModuleSpec.create(ResNet34, num_kp=256),
                "observation->image->wrist": ModuleSpec.create(ResNet34, num_kp=256),
                "observation->state": None,
            },
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [1024, 512, 512], activate_final=False), flatten_time=True
            ),
        ),
        action_head=ModuleSpec.create(
            DDPMActionHead,
            model=ModuleSpec.create(
                ConditionalUnet1D, down_features=(256, 512, 1024), mid_layers=2, time_features=256, kernel_size=5
            ),
            clip_sample=1.0,
            timesteps=100,
            variance_type="fixed_small",
            action_dim=10,
            action_horizon=16,
            num_noise_samples=4,
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=1000,
        decay_steps=400000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw, weight_decay=1e-5)

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=300000,
            log_freq=500,
            val_freq=5000,
            save_freq=50000,
            val_steps=25,
            exec_horizon=8,
            seed=seed,
        )
    )
