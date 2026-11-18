"""Play one Set 4 match with the historical MuZero policy. See markdown/full_game.md."""

import argparse
from pathlib import Path

import torch

import config as model_config
from Models.MCTS_torch import MCTS
from Models.MuZero_torch_agent import MuZeroNetwork
from Simulator import collect_episode, parallel_env, random_policy


def select_action(search, observation):
    mask = observation["action_mask"]
    batch = {
        "observations": {name: value[None] for name, value in observation["observations"].items()},
        "action_mask": mask[None],
    }
    labels, _, _, _ = search.policy(batch)
    # Legacy move labels reverse source and destination. The policy index matches the simulator mask.
    action = search.default_string_mapping[0].index(labels[0])
    if not mask[action]:
        raise ValueError(f"MuZero selected masked action {action}")
    return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Legacy checkpoint containing model_state_dict")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="PyTorch inference device")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple Metal (MPS) is unavailable in this PyTorch installation")
    model_config.DEVICE = args.device
    torch.set_num_threads(1)
    settings = model_config.ModelConfig()
    network = MuZeroNetwork(settings).to(args.device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(checkpoint["model_state_dict"])
    print("MuZero weights:", args.checkpoint or "untrained (random initialization)", flush=True)
    print("Network device:", next(network.parameters()).device, flush=True)
    search = MCTS(network.eval(), settings)

    def policy(observation, info, agent, env):
        if agent == "player_0":
            return select_action(search, observation)
        return random_policy(observation, info, agent, env)

    env = parallel_env()
    try:
        game = collect_episode(env, policy_fn=policy)
    finally:
        env.close()
    print("MuZero decisions:", len(game.transitions["player_0"]))
    print("Final placements:", game.placements)


if __name__ == "__main__":
    main()
