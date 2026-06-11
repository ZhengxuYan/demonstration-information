from typing import Dict

import tensorflow as tf

from openx.data.utils import RobotType, StateEncoding


def robomimic_transport_dataset_transform(ep: Dict):
    """Transform bimanual transport RoboMimic episodes for DemInf.

    The transport dataset stores 14D actions for two robots. Keep that action
    vector intact under MISC so DemInf VAEs score the full action, instead of
    incorrectly interpreting it as a single-robot 7D action.
    """

    observation = {
        "image": {
            "agent": ep["observation"]["agent_image"],
            "wrist": ep["observation"]["wrist_image"],
        },
        "state": {
            StateEncoding.MISC: ep["observation"]["state"]["transport"],
        },
    }

    action = {
        "desired_delta": {
            StateEncoding.MISC: tf.cast(ep["action"], tf.float32),
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    ep["ep_idx"] = ep["episode_metadata"]["ep_idx"]
    ep["quality_score"] = ep["episode_metadata"]["quality_score"]
    return ep
