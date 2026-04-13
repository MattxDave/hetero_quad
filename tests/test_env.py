from __future__ import annotations

import numpy as np

from sim import make_env


def _hold_action(env, agent: str) -> dict:
    state = env._states[agent]
    return {"type": 4, "target": np.array([state.x, state.y], dtype=np.float32)}


def _action_dict(env) -> dict[str, dict]:
    return {agent: _hold_action(env, agent) for agent in ["spot", "ghost"]}


# 1. Env builds for all 3 stages
def test_env_builds_all_stages() -> None:
    for stage in (1, 2, 3):
        env = make_env(stage)
        obs, infos = env.reset(seed=0)
        assert set(obs.keys()) == {"spot", "ghost"}
        env.close()


# 2. reset returns correct shapes  (10 + 4 + 3N + 100)
def test_reset_observation_shapes() -> None:
    env = make_env(2)  # N=3
    obs, _ = env.reset(seed=7)
    expected_dim = 10 + 4 + 3 * 3 + 100  # 123
    for agent in ("spot", "ghost"):
        assert obs[agent].shape == (expected_dim,)
        assert obs[agent].dtype == np.float32
    env.close()


# 3. step returns 5-tuple with correct keys
def test_step_returns_parallel_tuple() -> None:
    env = make_env(1)
    env.reset(seed=3)
    out = env.step(_action_dict(env))
    assert len(out) == 5
    obs, rewards, terms, truncs, infos = out
    for d in (obs, rewards, terms, truncs, infos):
        assert set(d.keys()) == {"spot", "ghost"}
    env.close()


# 4. 100 random steps — no crash, no NaN
def test_random_steps_no_nan() -> None:
    env = make_env(2)
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)
    for _ in range(100):
        actions = {
            agent: {
                "type": int(rng.integers(0, 5)),
                "target": np.array(
                    [rng.uniform(0, env.world.width), rng.uniform(0, env.world.height)],
                    dtype=np.float32,
                ),
            }
            for agent in ["spot", "ghost"]
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        for agent in ["spot", "ghost"]:
            assert np.isfinite(rewards[agent])
            assert np.all(np.isfinite(obs[agent]))
        if all(terms.values()) or all(truncs.values()):
            obs, _ = env.reset(seed=11)
    env.close()


# 5. Ghost INTERACT mask always 0
def test_action_mask_ghost_interact_always_zero() -> None:
    env = make_env(1)
    _, infos = env.reset(seed=0)
    assert infos["ghost"]["action_mask"][2] == 0.0
    _, _, _, _, infos = env.step(_action_dict(env))
    assert infos["ghost"]["action_mask"][2] == 0.0
    env.close()


# 6. Spot INTERACT mask flips when near revealed target
def test_action_mask_spot_interact_near_target() -> None:
    env = make_env(1)
    env.reset(seed=0)
    env._states["spot"].x = 0.5
    env._states["spot"].y = 0.5
    assert env._compute_action_mask("spot")[2] == 0.0
    t0 = env.tasks.targets[0]
    t0.revealed = True
    t0.interacted = False
    env._states["spot"].x = t0.x
    env._states["spot"].y = t0.y
    assert env._compute_action_mask("spot")[2] == 1.0
    env.close()


# 7. Spot INTERACT marks target done
def test_spot_interact_marks_target_done() -> None:
    env = make_env(1)
    env.reset(seed=22)
    t0 = env.tasks.targets[0]
    t0.revealed = True
    t0.interacted = False
    env._states["spot"].x = t0.x
    env._states["spot"].y = t0.y
    actions = _action_dict(env)
    actions["spot"] = {"type": 2, "target": np.array([t0.x, t0.y], dtype=np.float32)}
    env.step(actions)
    assert t0.interacted
    env.close()


# 8. Ghost INTERACT has no effect
def test_ghost_interact_has_no_effect() -> None:
    env = make_env(1)
    env.reset(seed=21)
    t0 = env.tasks.targets[0]
    t0.revealed = True
    t0.interacted = False
    env._states["ghost"].x = t0.x
    env._states["ghost"].y = t0.y
    actions = _action_dict(env)
    actions["ghost"] = {"type": 2, "target": np.array([t0.x, t0.y], dtype=np.float32)}
    env.step(actions)
    assert not t0.interacted
    env.close()


# 9. Deterministic resets
def test_deterministic_reset() -> None:
    env = make_env(2)
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    for agent in ["spot", "ghost"]:
        np.testing.assert_array_equal(obs1[agent], obs2[agent])
    env.close()


# 10. Episode terminates on success (all targets interacted)
def test_episode_terminates_on_all_interacted() -> None:
    env = make_env(1)  # 2 targets
    env.reset(seed=42)
    for target in env.tasks.targets:
        target.revealed = True
    terms_result = {}
    for target in env.tasks.targets:
        env._states["spot"].x = target.x
        env._states["spot"].y = target.y
        actions = _action_dict(env)
        actions["spot"] = {"type": 2, "target": np.array([target.x, target.y], dtype=np.float32)}
        _, _, terms_result, _, _ = env.step(actions)
    assert all(terms_result.values())
    env.close()


# 11. Episode terminates on battery death
def test_episode_terminates_on_battery_death() -> None:
    env = make_env(2)  # battery drain ON
    env.reset(seed=42)
    env._states["ghost"].battery = 1e-5
    actions = _action_dict(env)
    actions["ghost"] = {"type": 1, "target": np.array([10.0, 10.0], dtype=np.float32)}
    _, _, terms, _, _ = env.step(actions)
    assert all(terms.values())
    env.close()


# 12. Role metrics populated with finite floats
def test_role_metrics_populated() -> None:
    env = make_env(1)
    env.reset(seed=0)
    _, _, _, _, infos = env.step(_action_dict(env))
    for agent in ["spot", "ghost"]:
        rm = infos[agent]["role_metrics"]
        assert "scout_rate_ghost" in rm
        assert "interact_rate_spot" in rm
        assert np.isfinite(rm["scout_rate_ghost"])
        assert np.isfinite(rm["interact_rate_spot"])
    env.close()
