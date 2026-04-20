#!/usr/bin/env python3
"""Add a discrete binned Transformer BC head to the robomimic submodule.

This keeps the official robomimic checkout as a submodule while allowing this
project to run an experimental action-discretized policy on the cluster.
The patch is intentionally idempotent so Slurm jobs can apply it at startup.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BC_CONFIG = ROOT / "robomimic" / "robomimic" / "config" / "bc_config.py"
BC_ALGO = ROOT / "robomimic" / "robomimic" / "algo" / "bc.py"
POLICY_NETS = ROOT / "robomimic" / "robomimic" / "models" / "policy_nets.py"


def replace_once(path, old, new):
    text = path.read_text()
    if new in text:
        print(f"Already patched: {path}")
        return
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"Patched: {path}")


def patch_config():
    text = BC_CONFIG.read_text()
    if "self.algo.discrete.enabled" in text:
        print(f"Already patched: {BC_CONFIG}")
        return
    marker = "        # stochastic VAE policy settings\n"
    insert = """        # discrete binned policy settings
        self.algo.discrete.enabled = False              # whether to train a categorical policy over action bins
        self.algo.discrete.num_bins = 256               # number of uniform bins per action dimension
        self.algo.discrete.bin_type = \"uniform\"         # currently only uniform bins over [-1, 1]

"""
    if marker not in text:
        raise RuntimeError(f"Expected VAE settings marker not found in {BC_CONFIG}")
    BC_CONFIG.write_text(text.replace(marker, insert + marker, 1))
    print(f"Patched: {BC_CONFIG}")


def patch_policy_nets():
    old = """
class TransformerGMMActorNetwork(TransformerActorNetwork):
"""
    new = """
class TransformerDiscreteActorNetwork(TransformerActorNetwork):
    \"\"\"
    Transformer policy that predicts categorical logits over uniform action bins
    independently for each action dimension.
    \"\"\"
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation=\"gelu\",
        transformer_nn_parameter_for_timesteps=False,
        num_bins=64,
        bin_type=\"uniform\",
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        assert bin_type == \"uniform\", \"Only uniform action bins are currently supported\"
        self.num_bins = num_bins
        self.bin_type = bin_type
        super(TransformerDiscreteActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            transformer_embed_dim=transformer_embed_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_context_length=transformer_context_length,
            transformer_emb_dropout=transformer_emb_dropout,
            transformer_attn_dropout=transformer_attn_dropout,
            transformer_block_output_dropout=transformer_block_output_dropout,
            transformer_sinusoidal_embedding=transformer_sinusoidal_embedding,
            transformer_activation=transformer_activation,
            transformer_nn_parameter_for_timesteps=transformer_nn_parameter_for_timesteps,
            encoder_kwargs=encoder_kwargs,
            goal_shapes=goal_shapes,
        )

    def _get_output_shapes(self):
        return OrderedDict(logits=(self.ac_dim, self.num_bins))

    def output_shape(self, input_shape):
        mod = list(self.obs_shapes.keys())[0]
        T = input_shape[mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0,
                msg=\"TransformerDiscreteActorNetwork: input_shape inconsistent in temporal dimension\")
        return [T, self.ac_dim, self.num_bins]

    def forward_train(self, obs_dict, actions=None, goal_dict=None):
        if self._is_goal_conditioned:
            assert goal_dict is not None
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        outputs = MIMO_Transformer.forward(self, obs=obs_dict, goal=goal_dict)
        return outputs[\"logits\"]

    def forward(self, obs_dict, actions=None, goal_dict=None):
        logits = self.forward_train(obs_dict=obs_dict, actions=actions, goal_dict=goal_dict)
        bins = torch.argmax(logits, dim=-1)
        centers = torch.linspace(
            -1.0 + 1.0 / self.num_bins,
            1.0 - 1.0 / self.num_bins,
            self.num_bins,
            device=logits.device,
            dtype=logits.dtype,
        )
        return centers[bins]

    def _to_string(self):
        return \"action_dim={}, num_bins={}, bin_type={}\".format(self.ac_dim, self.num_bins, self.bin_type)


class TransformerGMMActorNetwork(TransformerActorNetwork):
"""
    replace_once(POLICY_NETS, old, new)


def patch_plain_policy_nets():
    text = POLICY_NETS.read_text()
    if "class DiscreteActorNetwork(ActorNetwork):" in text:
        print(f"Already patched: {POLICY_NETS}")
        return

    if "\nclass TransformerDiscreteActorNetwork" in text:
        marker = "\nclass TransformerDiscreteActorNetwork"
    else:
        marker = "\nclass TransformerGMMActorNetwork"

    insert = """
class DiscreteActorNetwork(ActorNetwork):
    \"\"\"
    Plain MLP policy that predicts categorical logits over uniform action bins
    independently for each action dimension.
    \"\"\"
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        num_bins=64,
        bin_type=\"uniform\",
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        assert bin_type == \"uniform\", \"Only uniform action bins are currently supported\"
        self.num_bins = num_bins
        self.bin_type = bin_type
        super(DiscreteActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        return OrderedDict(logits=(self.ac_dim, self.num_bins))

    def output_shape(self, input_shape=None):
        return [self.ac_dim, self.num_bins]

    def forward_train(self, obs_dict, goal_dict=None):
        return MIMO_MLP.forward(self, obs=obs_dict, goal=goal_dict)[\"logits\"]

    def forward(self, obs_dict, goal_dict=None):
        logits = self.forward_train(obs_dict=obs_dict, goal_dict=goal_dict)
        bins = torch.argmax(logits, dim=-1)
        centers = torch.linspace(
            -1.0 + 1.0 / self.num_bins,
            1.0 - 1.0 / self.num_bins,
            self.num_bins,
            device=logits.device,
            dtype=logits.dtype,
        )
        return centers[bins]

    def _to_string(self):
        return \"action_dim={}, num_bins={}, bin_type={}\".format(self.ac_dim, self.num_bins, self.bin_type)

"""
    if marker not in text:
        raise RuntimeError(f"Expected policy-net insertion marker not found in {POLICY_NETS}")
    POLICY_NETS.write_text(text.replace(marker, insert + marker, 1))
    print(f"Patched: {POLICY_NETS}")


def patch_bc_algo():
    old_factory = """    gmm_enabled = (\"gmm\" in algo_config and algo_config.gmm.enabled)
    vae_enabled = (\"vae\" in algo_config and algo_config.vae.enabled)
"""
    new_factory = """    gmm_enabled = (\"gmm\" in algo_config and algo_config.gmm.enabled)
    discrete_enabled = (\"discrete\" in algo_config and algo_config.discrete.enabled)
    vae_enabled = (\"vae\" in algo_config and algo_config.vae.enabled)
"""
    replace_once(BC_ALGO, old_factory, new_factory)

    old_branch = """    if gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
"""
    new_branch = """    if discrete_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_Discrete, {}
        else:
            raise NotImplementedError
    elif gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
"""
    text = BC_ALGO.read_text()
    if "    if discrete_enabled:\n" in text:
        print(f"Already patched: {BC_ALGO}")
    else:
        replace_once(BC_ALGO, old_branch, new_branch)

    old_class = """
class BC_Transformer_GMM(BC_Transformer):
"""
    new_class = """
class BC_Transformer_Discrete(BC_Transformer):
    \"\"\"
    BC training with a Transformer categorical policy over uniform action bins.
    \"\"\"
    def _create_networks(self):
        assert self.algo_config.discrete.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets[\"policy\"] = PolicyNets.TransformerDiscreteActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_bins=self.algo_config.discrete.num_bins,
            bin_type=self.algo_config.discrete.bin_type,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _actions_to_bins(self, actions):
        num_bins = self.algo_config.discrete.num_bins
        clipped = torch.clamp(actions, -1.0, 1.0 - 1e-6)
        bins = torch.floor((clipped + 1.0) * 0.5 * num_bins).long()
        return torch.clamp(bins, 0, num_bins - 1)

    def _forward_training(self, batch, epoch=None):
        TensorUtils.assert_size_at_dim(
            batch[\"obs\"],
            size=(self.context_length),
            dim=1,
            msg=\"Error: expect temporal dimension of obs batch to match transformer context length {}\".format(self.context_length),
        )

        logits = self.nets[\"policy\"].forward_train(
            obs_dict=batch[\"obs\"],
            actions=None,
            goal_dict=batch[\"goal_obs\"],
        )
        if not self.supervise_all_steps:
            logits = logits[:, -1, :, :]

        target_bins = self._actions_to_bins(batch[\"actions\"])
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_bins.reshape(-1),
            reduction=\"none\",
        ).reshape(target_bins.shape)
        nll = ce.sum(dim=-1)

        return OrderedDict(
            logits=logits,
            target_bins=target_bins,
            log_probs=-nll,
        )

    def _compute_losses(self, predictions, batch):
        action_loss = -predictions[\"log_probs\"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log[\"Loss\"] = info[\"losses\"][\"action_loss\"].item()
        log[\"Log_Likelihood\"] = info[\"losses\"][\"log_probs\"].item()
        if \"policy_grad_norms\" in info:
            log[\"Policy_Grad_Norms\"] = info[\"policy_grad_norms\"]
        return log


class BC_Transformer_GMM(BC_Transformer):
"""
    replace_once(BC_ALGO, old_class, new_class)


def patch_plain_bc_algo():
    text = BC_ALGO.read_text()
    old_branch = """    if discrete_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_Discrete, {}
        else:
            raise NotImplementedError
"""
    new_branch = """    if discrete_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_Discrete, {}
        else:
            algo_class, algo_kwargs = BC_Discrete, {}
"""
    if new_branch in text:
        print(f"Already patched: {BC_ALGO}")
    elif old_branch in text:
        BC_ALGO.write_text(text.replace(old_branch, new_branch, 1))
        print(f"Patched: {BC_ALGO}")
    else:
        raise RuntimeError(f"Expected discrete factory branch not found in {BC_ALGO}")

    text = BC_ALGO.read_text()
    if "class BC_Discrete(BC_Gaussian):" in text:
        print(f"Already patched: {BC_ALGO}")
        return

    marker = "\nclass BC_Transformer_Discrete"
    if marker not in text:
        marker = "\nclass BC_Transformer_GMM"

    insert = """
class BC_Discrete(BC_Gaussian):
    \"\"\"
    Plain BC training with a categorical policy over uniform action bins.
    \"\"\"
    def _create_networks(self):
        assert self.algo_config.discrete.enabled

        self.nets = nn.ModuleDict()
        self.nets[\"policy\"] = PolicyNets.DiscreteActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_bins=self.algo_config.discrete.num_bins,
            bin_type=self.algo_config.discrete.bin_type,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def _actions_to_bins(self, actions):
        num_bins = self.algo_config.discrete.num_bins
        clipped = torch.clamp(actions, -1.0, 1.0 - 1e-6)
        bins = torch.floor((clipped + 1.0) * 0.5 * num_bins).long()
        return torch.clamp(bins, 0, num_bins - 1)

    def _forward_training(self, batch):
        logits = self.nets[\"policy\"].forward_train(
            obs_dict=batch[\"obs\"],
            goal_dict=batch[\"goal_obs\"],
        )
        target_bins = self._actions_to_bins(batch[\"actions\"])
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_bins.reshape(-1),
            reduction=\"none\",
        ).reshape(target_bins.shape)
        nll = ce.sum(dim=-1)

        return OrderedDict(
            logits=logits,
            target_bins=target_bins,
            log_probs=-nll,
        )


"""
    if marker not in text:
        raise RuntimeError(f"Expected BC insertion marker not found in {BC_ALGO}")
    BC_ALGO.write_text(text.replace(marker, insert + marker, 1))
    print(f"Patched: {BC_ALGO}")


def main():
    patch_config()
    patch_policy_nets()
    patch_bc_algo()
    patch_plain_policy_nets()
    patch_plain_bc_algo()


if __name__ == "__main__":
    main()
