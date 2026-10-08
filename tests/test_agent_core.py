"""Fast tests for the coding-agent benchmark core (no simulator, no network).

Covers the wire protocol, the policy loader, the episode loop's scoring rule,
the server's seed refusal and the ``bigym-agent`` dispatcher.
"""

import socket
import threading

import numpy as np
import pytest

from bigym.loco.agent import cli, wire
from bigym.loco.agent.episode import load_policy, run_episode
from bigym.loco.agent.server import EVAL_SEED_HI, EVAL_SEED_LO, check_reset_seed

POLICY_SRC = """
import numpy as np


class Policy:
    def reset(self, obs, tools):
        self.hold = np.asarray(tools.hold_action(), dtype=np.float32)

    def act(self, obs, tools):
        return self.hold
"""


class FakeEnv:
    """Minimal stand-in for the client's Env: no MuJoCo, scripted outcome."""

    def __init__(self, steps=3, reward_at_end=1.0, fell=False, termination="success"):
        self.info = {"action_dim": 20, "task": "fake", "time_limit": 10}
        self.steps = steps
        self.reward_at_end = reward_at_end
        self.fell = fell
        self.termination = termination
        self.t = 0
        self.actions = []

    def _obs(self):
        return {
            "t": self.t,
            "time_limit": 10,
            "fell": self.fell,
            "low_dim_obs": np.zeros(50),
        }

    def reset(self, seed):
        self.t = 0
        self.seed = seed
        return self._obs()

    def step(self, raw):
        self.actions.append(np.asarray(raw))
        self.t += 1
        done = self.t >= self.steps
        reward = self.reward_at_end if done else 0.0
        return (
            self._obs(),
            reward,
            done,
            {"termination": self.termination if done else None},
        )

    def hold_action(self):
        return np.zeros(20, dtype=np.float32)


def _socketpair():
    a, b = socket.socketpair()
    return a, b


def test_wire_round_trip_over_socketpair():
    a, b = _socketpair()
    msg = {
        "op": "step",
        "action": np.arange(4, dtype=np.float32),
        "n": np.int64(7),
        "ok": True,
    }
    got = {}

    def reader():
        got["msg"] = wire.recv(b)

    t = threading.Thread(target=reader)
    t.start()
    wire.send(a, msg)
    t.join(timeout=5)
    a.close()
    b.close()
    assert got["msg"] == {
        "op": "step",
        "action": [0.0, 1.0, 2.0, 3.0],
        "n": 7,
        "ok": True,
    }


def test_wire_refuses_an_oversized_message():
    a, b = _socketpair()
    a.sendall((wire.MAX_MSG + 1).to_bytes(4, "big"))
    with pytest.raises(ValueError):
        wire.recv(b)
    a.close()
    b.close()


def test_load_policy_and_missing_policy_class(tmp_path):
    good = tmp_path / "policy.py"
    good.write_text(POLICY_SRC)
    policy = load_policy(good)
    assert hasattr(policy, "reset") and hasattr(policy, "act")

    bad = tmp_path / "not_a_policy.py"
    bad.write_text("X = 1\n")
    with pytest.raises(AttributeError, match="must define a class Policy"):
        load_policy(bad)


def test_run_episode_scores_a_success(tmp_path):
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(POLICY_SRC)
    env = FakeEnv(steps=3, reward_at_end=1.0)
    rec = run_episode(env, load_policy(policy_path), seed=7)
    assert set(rec) >= {
        "seed",
        "success",
        "length",
        "reward",
        "termination",
        "fell",
        "wall_s",
        "final_obs",
    }
    assert rec["seed"] == 7
    assert rec["success"] == 1
    assert rec["length"] == 3
    assert rec["reward"] == pytest.approx(1.0)
    assert rec["termination"] == "success"
    assert rec["fell"] is False
    assert "low_dim_obs" not in rec["final_obs"]
    assert len(env.actions) == 3


def test_run_episode_counts_a_fall_as_failure(tmp_path):
    """A rewarded episode in which the robot fell is a failure (protocol v1)."""
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(POLICY_SRC)
    rec = run_episode(
        FakeEnv(steps=2, reward_at_end=1.0, fell=True), load_policy(policy_path), 3
    )
    assert rec["success"] == 0
    assert rec["fell"] is True
    assert rec["termination"] == "fell"


def test_run_episode_below_the_success_threshold(tmp_path):
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(POLICY_SRC)
    rec = run_episode(FakeEnv(steps=2, reward_at_end=0.1), load_policy(policy_path), 3)
    assert rec["success"] == 0


def test_run_episode_stops_at_the_time_limit(tmp_path):
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(POLICY_SRC)
    rec = run_episode(
        FakeEnv(steps=10**6, reward_at_end=1.0), load_policy(policy_path), 1
    )
    assert rec["length"] == 10  # obs["time_limit"]
    assert rec["termination"] == "timeout"
    assert rec["success"] == 0


@pytest.mark.parametrize("seed", [EVAL_SEED_LO, EVAL_SEED_LO + 42, EVAL_SEED_HI - 1])
def test_server_refuses_the_hidden_seed_block(seed):
    with pytest.raises(ValueError, match="not allowed"):
        check_reset_seed(seed)
    with pytest.raises(ValueError, match="not allowed"):
        check_reset_seed(seed, allowed_seeds=frozenset({seed}))
    # the in-the-loop evaluation is the one caller that may use them
    assert check_reset_seed(seed, allow_eval_seeds=True) == seed


def test_the_evaluation_seed_block_is_pinned_everywhere_it_is_used():
    """Episode i of the 100 uses 620000 + i, and the server gates [620000, 630000).

    Every published score was produced on this block; moving it, or letting
    one caller drift from the protocol's definition, breaks comparability.
    """
    from bigym.loco.agent import evaluate, replay
    from bigym.loco.agent.inloop import run_inloop
    from bigym.loco.eval import protocol
    from bigym.vr.viewer import agent_cells

    assert protocol.EVAL_SEED_BASE == 620000
    assert protocol.EVAL_EPISODES == 100
    assert protocol.eval_seeds() == list(range(620000, 620100))
    assert protocol.EVAL_SEED_RESERVED == (620000, 630000)
    assert (EVAL_SEED_LO, EVAL_SEED_HI) == protocol.EVAL_SEED_RESERVED
    assert evaluate.EVAL_SEED_BASE == protocol.EVAL_SEED_BASE
    assert cli.EvaluateConfig.seed_base == protocol.EVAL_SEED_BASE
    assert run_inloop.EVAL_SEED_BASE == protocol.EVAL_SEED_BASE
    assert agent_cells.EVAL_SEED_LO == protocol.EVAL_SEED_BASE
    assert cli.ReplayConfig.seeds == agent_cells.DEFAULT_REPLAY_SEEDS
    assert replay.parse_seeds(cli.ReplayConfig.seeds) == protocol.eval_seeds(5)


def test_cli_defaults_match_the_modules_they_configure():
    """The settings classes repeat a few module constants (see bigym.loco.agent.cli)."""
    from bigym.loco.agent import server

    assert cli.ServeConfig.train_seeds == server.MAX_TRAIN_SEED


def test_server_seed_check_accepts_development_seeds():
    assert check_reset_seed(0) == 0
    assert check_reset_seed(59) == 59
    with pytest.raises(ValueError):
        check_reset_seed(60)
    with pytest.raises(ValueError):
        check_reset_seed(-1)
    assert check_reset_seed(7, allowed_seeds=frozenset({7, 9})) == 7
    with pytest.raises(ValueError):
        check_reset_seed(8, allowed_seeds=frozenset({7, 9}))


def test_cli_help_lists_the_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in cli.CONFIGS:
        assert name in out


def test_cli_unknown_subcommand_lists_the_known_ones(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2
    err = " ".join(capsys.readouterr().err.split())
    assert "nope" in err
    for name in cli.CONFIGS:
        assert name in err


def test_cli_every_subcommand_runs_something():
    for name, config in cli.CONFIGS.items():
        assert "run" in vars(config), name


def test_cli_runs_as_a_module():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "bigym.loco.agent", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert "--ledger-dir" in out.stdout


def test_cli_subcommand_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "--help"])
    assert exc.value.code == 0
    assert "--policy" in capsys.readouterr().out


def test_importing_the_package_pulls_in_no_optional_dependency():
    """The agent extra (cv2, scipy, mink) must not be imported by the package."""
    import subprocess
    import sys

    code = (
        "import bigym.loco.agent, sys; "
        "print([m for m in sys.modules if m.startswith(('cv2', 'scipy', 'mink'))])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_templates_and_docker_files_ship():
    from pathlib import Path

    import bigym.loco.agent as agent_pkg

    root = Path(agent_pkg.__file__).parent
    for name in (
        "PROMPT.md",
        "PROMPT_INLOOP.md",
        "PROMPT_INLOOP_IMAGES.md",
        "api_strict.md",
        "api_tools.md",
        "policy_template.py",
    ):
        assert (root / "templates" / name).is_file(), name
    for name in (
        "Dockerfile.codex",
        "Dockerfile.claude",
        "codex_system_config.toml",
    ):
        assert (root / "docker" / name).is_file(), name
    strict = (root / "templates" / "api_strict.md").read_text()
    for withheld in ("camera_info", "pixel_to_ray", "tools.ik", "base_pos"):
        assert withheld not in strict, withheld
    tools_doc = (root / "templates" / "api_tools.md").read_text()
    for offered in ("camera_info", "pixel_to_ray", "tools.ik", "base_pos"):
        assert offered in tools_doc, offered


def test_prepare_sock_dir_short_and_long(tmp_path):
    """A short sandbox binds in place; a deep one gets a linked short dir."""
    import os
    import shutil
    from pathlib import Path

    from bigym.loco.agent import server

    short = tmp_path / "sb"
    assert server.prepare_sock_dir(short) == short / "sock"
    assert (short / "sock").is_dir() and not (short / "sock").is_symlink()

    deep = tmp_path / ("d" * 60) / ("e" * 60) / "sandbox"
    deep.mkdir(parents=True)
    real = server.prepare_sock_dir(deep)
    assert (deep / "sock").is_symlink()
    assert real == Path(os.readlink(deep / "sock"))
    assert len(str(real / "w99.sock").encode()) <= server.SOCK_PATH_LIMIT
    # idempotent: a second call returns the same directory
    assert server.prepare_sock_dir(deep) == real
    (deep / "sock" / "w0.sock").write_text("")  # visible through the link
    assert (real / "w0.sock").exists()
    shutil.rmtree(real)  # the server removes its temp directory on shutdown
