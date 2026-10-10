"""RADDTLinear actor and MLP critic used by the SpinningUp DDPG experiment."""

import sys

import torch
import torch.nn as nn

sys.path.append('./src/')
sys.path.append('./src/DT/')

from treeFunc import readTreePath, treePredYhatLinear
from ddpg.core import MLPQFunction


class RADDTLinearActor(torch.nn.Module):
    def __init__(self, treeDepth, obs_dim, act_dim, act_limit, scale, warmStart=None):
        super().__init__()
        self.depth = treeDepth
        self.featNum = obs_dim
        self.branchNodeNum = 2 ** self.depth - 1
        self.scale = scale
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.linear1 = torch.nn.Linear(self.featNum, self.branchNodeNum)

        if warmStart is None:
            aTensor = torch.rand(self.linear1.weight.shape, dtype=torch.float32) * 2.0 - 1.0
            bTensor = torch.rand(self.linear1.bias.shape, dtype=torch.float32) * 2.0 - 1.0
            kTensor = torch.rand((2 ** self.depth, act_dim, obs_dim), dtype=torch.float32) * 2.0 - 1.0
            hTensor = torch.rand((2 ** self.depth, act_dim), dtype=torch.float32) * 2.0 - 1.0
        else:
            aTensor = torch.as_tensor(warmStart["a"], dtype=torch.float32)
            bTensor = -torch.as_tensor(warmStart["b"], dtype=torch.float32)
            kTensor = torch.as_tensor(warmStart["k"], dtype=torch.float32)
            hTensor = torch.as_tensor(warmStart["h"], dtype=torch.float32)

        self.linear1.weight = torch.nn.Parameter(aTensor.clone().detach().requires_grad_(True))
        self.linear1.bias = torch.nn.Parameter(bTensor.clone().detach().requires_grad_(True))
        self.k_leafSlope = torch.nn.Parameter(kTensor.clone().detach().requires_grad_(True))
        self.h_leafIntercept = torch.nn.Parameter(hTensor.clone().detach().requires_grad_(True))

        indices_flags_dict = readTreePath(self.depth)
        self.register_buffer(
            "indices_tensor",
            indices_flags_dict["D" + str(self.depth)]["indices_tensor"],
        )
        self.register_buffer(
            "flags_tensor",
            indices_flags_dict["D" + str(self.depth)]["flags_tensor"],
        )

    def forward(self, obs):
        squeeze_output = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True

        branch_output = self.linear1(obs)
        relu_b_minus_ax = torch.relu(-branch_output)
        relu_ax_minus_b = torch.relu(branch_output)
        indicators_stack = torch.stack((relu_b_minus_ax, relu_ax_minus_b))
        selected_indicators = indicators_stack[self.flags_tensor, :, self.indices_tensor]
        path_distances = selected_indicators.sum(dim=1).transpose(0, 1)
        leaf_weights = torch.nn.functional.softmin(
            self.scale * path_distances,
            dim=1,
        )
        if self.depth > 4:
            leaf_weights = torch.nn.functional.threshold(
                leaf_weights,
                1.0 / (2 ** self.depth),
                0.0,
            )

        batch_size = obs.shape[0]
        leaf_num = self.k_leafSlope.shape[0]
        obs_exp = obs.unsqueeze(1).expand(-1, leaf_num, -1)
        k_exp = self.k_leafSlope.unsqueeze(0).expand(batch_size, -1, -1, -1)
        obs_exp = obs_exp.unsqueeze(2)
        leaf_yhat = torch.matmul(k_exp, obs_exp.transpose(-1, -2)).squeeze(-1)
        leaf_yhat = leaf_yhat + self.h_leafIntercept.unsqueeze(0)
        yhat = (leaf_weights.unsqueeze(-1) * leaf_yhat).sum(dim=1)
        yhat = torch.tanh(yhat)

        if squeeze_output:
            yhat = yhat.squeeze(0)

        return self.act_limit * yhat

    def hardAct(self, obs):
        with torch.no_grad():
            squeeze_output = False
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
                squeeze_output = True

            aTrained = self.linear1.weight.clone().detach()
            bTrained = self.linear1.bias.clone().detach() * (-1.0)
            kTrained = self.k_leafSlope.clone().detach()
            hTrained = self.h_leafIntercept.clone().detach()
            treeTrained = {"a": aTrained, "b": bTrained, "k": kTrained, "h": hTrained}

            yhat = torch.tanh(treePredYhatLinear(obs, self.depth, treeTrained))
            yhat = self.act_limit * yhat
            if squeeze_output:
                yhat = yhat.squeeze(0)
            return yhat


class RADDTLinearActorMLPCritic(nn.Module):
    def __init__(self, observation_space, action_space, treeDepth, scale, warmStart,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")

        self.pi = RADDTLinearActor(treeDepth, obs_dim, act_dim, act_limit, scale, warmStart)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


DTActorMLPCritic = RADDTLinearActorMLPCritic


if __name__ == "__main__":
    torch.manual_seed(0)

    import gym

    observation_space = gym.make('Pendulum-v1').observation_space
    action_space = gym.make('Pendulum-v1').action_space

    treeDepth = 8
    scale = 1.0
    warmStart = None
    actor_critic = RADDTLinearActorMLPCritic(
        observation_space,
        action_space,
        treeDepth,
        scale,
        warmStart,
    )

    x = torch.rand(5, observation_space.shape[0])
    output = actor_critic.pi(x)
    print("Output shape:", output.shape)
    print("Output:", output)

    for name, param in actor_critic.pi.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
