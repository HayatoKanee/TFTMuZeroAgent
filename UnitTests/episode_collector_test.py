from Simulator.generators.episode_collector import collect_episode
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env
from examples.default_agent_vs_random import mixed_policy


def test_collect_episode_saves_placements(tmp_path):
    env = parallel_env(TFTConfig(num_players=8))
    agents = list(env.possible_agents)
    recorder = collect_episode(env, max_steps=8)
    env.close()

    path = recorder.save(tmp_path / "episode_00000.npz")
    assert path.exists()
    assert set(recorder.placements.keys()) == set(agents)
    assert all(place is not None for place in recorder.placements.values())
    assert any(recorder.transitions[agent] for agent in recorder.agents)


def test_full_episode_finishes_with_the_surviving_player_first():
    env = parallel_env(TFTConfig(max_actions_per_round=1))

    def policy(observation, info, agent, env):
        assert agent in env.agents
        if agent not in ("player_0", "player_1") or (agent == "player_1" and len(env.agents) == 2):
            info["player"].health = 0
        return mixed_policy(observation, info, agent, env)

    try:
        recorder = collect_episode(env, policy_fn=policy)
        assert recorder.placements["player_0"] == 1
        assert recorder.placements["player_1"] == 2
        assert sorted(recorder.placements.values()) == list(range(1, 9))
    finally:
        env.close()


def test_terminal_placements_order_multiple_survivors_by_health(monkeypatch):
    env = parallel_env(TFTConfig(max_actions_per_round=1))
    try:
        _, infos = env.reset(seed=7)
        for seat, agent in enumerate(env.possible_agents, start=1):
            infos[agent]["player"].health = 10 * seat
        monkeypatch.setattr(env.unwrapped, "game_over", lambda: True)
        _, _, _, _, infos = env.step({agent: [0, 0, 0] for agent in env.agents})
        assert [infos[agent]["placement"] for agent in env.possible_agents] == list(range(8, 0, -1))
    finally:
        env.close()
