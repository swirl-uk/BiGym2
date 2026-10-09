r"""Train an ACT-style policy on BiGym demonstrations through the paper's API.

This is the reference usage from the paper, verbatim in shape::

    from bigym.loco import make_gym

    env = make_gym("move_plate")
    agent = Agent(env.observation_space, env.action_space)
    agent.ingest(env.get_demos(60))

    obs, _ = env.reset()
    for _ in range(100_000):
        action = agent.act(obs)
        nxt, reward, term, trunc, _ = env.step(action)
        agent.observe(obs, action, reward, nxt, term, trunc)
        agent.update()
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
    env.close()

``Agent`` here is a compact Action Chunking Transformer (ACT: a CNN per
camera + state MLP feeding a transformer decoder that emits a chunk of K
future actions, trained with L1 on the demonstrations, executed with
temporal ensembling). It is imitation learning only: ``observe`` keeps the
online transitions but ``update`` trains on the demonstrations, exactly as
ACT does. The point of the file is to exercise the full pipeline end to end
(dataset download, ``get_demos``, gymnasium loop, evaluation protocol) on a
small budget, not to reproduce a paper number::

    MUJOCO_GL=egl uv run --with torch python examples/train_act.py \\
        --task reach_target_single --demos 10 --steps 2000 --eval-episodes 3

torch is the only thing beyond the core install. ``--with`` adds it for this
run without touching the project environment; pick the torch build for your
hardware as usual. On macOS drop ``MUJOCO_GL=egl``.
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from bigym.loco import make_gym


class Agent:
    """ACT-style behaviour-cloning agent behind the paper's four-method API."""

    def __init__(
        self,
        observation_space,
        action_space,
        *,
        chunk=20,
        hidden=256,
        lr=1e-4,
        batch=32,
    ):
        """Build the policy for the env's observation and action spaces."""
        import torch
        import torch.nn as nn

        self.torch = torch
        self.chunk = int(chunk)
        self.batch = int(batch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rgb_shape = observation_space["rgb"].shape  # (cams, 3 * stack, H, W)
        state_dim = observation_space["state"].shape[0]
        self.action_dim = action_space.shape[0]
        self.action_low = torch.as_tensor(action_space.low, device=self.device)
        self.action_high = torch.as_tensor(action_space.high, device=self.device)
        cams, channels = rgb_shape[0], rgb_shape[1]

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(channels, 32, 8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(64 * 16, hidden),
                    nn.LayerNorm(hidden),
                )
                self.state = nn.Sequential(
                    nn.Linear(state_dim, hidden), nn.LayerNorm(hidden)
                )
                self.cam_embed = nn.Parameter(torch.zeros(cams, hidden))

            def forward(self, rgb, state):
                # rgb: (B, cams, C, H, W) uint8 -> tokens (B, cams + 1, hidden)
                b = rgb.shape[0]
                x = rgb.flatten(0, 1).float() / 255.0
                tokens = self.cnn(x).view(b, cams, -1) + self.cam_embed
                return torch.cat([tokens, self.state(state)[:, None]], dim=1)

        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                layer = nn.TransformerDecoderLayer(
                    hidden, 4, hidden * 2, dropout=0.1, batch_first=True
                )
                self.decoder = nn.TransformerDecoder(layer, num_layers=2)
                self.queries = nn.Parameter(torch.randn(chunk, hidden) * 0.02)
                self.head = nn.Linear(hidden, action_space.shape[0])

            def forward(self, rgb, state):
                memory = self.encoder(rgb, state)
                queries = self.queries[None].expand(rgb.shape[0], -1, -1)
                return torch.tanh(self.head(self.decoder(queries, memory)))

        self.policy = Policy().to(self.device)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.rgb: np.ndarray | None = None  # demo frames, (N, cams, C, H, W)
        self.state: np.ndarray | None = None
        self.chunks: np.ndarray | None = (
            None  # (N, chunk, A), padded with the last action
        )
        self.online = collections.deque(maxlen=10_000)
        self.ensemble: collections.deque = collections.deque()
        self.losses: list[float] = []

    # -- paper API -------------------------------------------------------
    def ingest(self, demos):
        """Index every demo frame with the chunk of actions that follows it."""
        rgb, state, chunks = [], [], []
        for demo in demos:
            actions = demo["action"]
            for t in range(len(actions)):
                window = actions[t : t + self.chunk]
                if len(window) < self.chunk:
                    window = np.concatenate(
                        [window, np.repeat(window[-1:], self.chunk - len(window), 0)]
                    )
                rgb.append(demo["obs"]["rgb"][t])
                state.append(demo["obs"]["state"][t])
                chunks.append(window)
        self.rgb = np.stack(rgb)
        self.state = np.stack(state).astype(np.float32)
        self.chunks = np.stack(chunks).astype(np.float32)
        print(f"ingested {len(demos)} demos -> {len(self.rgb)} training frames")

    def act(self, obs):
        """Return the action for one gymnasium observation (temporal ensembling)."""
        torch = self.torch
        self.policy.eval()
        with torch.no_grad():
            rgb = torch.as_tensor(obs["rgb"], device=self.device)[None]
            state = torch.as_tensor(
                obs["state"], device=self.device, dtype=torch.float32
            )[None]
            chunk = self.policy(rgb, state)[0].cpu().numpy()
        # Temporal ensembling: average every chunk's prediction for this step,
        # exponentially favouring older (more context-committed) chunks.
        self.ensemble.append(chunk)
        preds = [c[i] for i, c in enumerate(reversed(self.ensemble)) if i < self.chunk]
        weights = np.exp(-0.01 * np.arange(len(preds)))[::-1]
        action = np.average(np.stack(preds), axis=0, weights=weights)
        if len(self.ensemble) >= self.chunk:
            self.ensemble.popleft()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        """Record the online transition; reset the ensemble at episode ends."""
        self.online.append((action, reward, terminated, truncated))
        if terminated or truncated:
            self.ensemble.clear()

    def update(self):
        """One L1 gradient step on a demonstration batch (behaviour cloning)."""
        if self.rgb is None:
            return
        assert self.state is not None and self.chunks is not None  # set with rgb
        torch = self.torch
        self.policy.train()
        idx = np.random.randint(0, len(self.rgb), self.batch)
        rgb = torch.as_tensor(self.rgb[idx], device=self.device)
        state = torch.as_tensor(self.state[idx], device=self.device)
        target = torch.as_tensor(self.chunks[idx], device=self.device)
        loss = torch.nn.functional.l1_loss(self.policy(rgb, state), target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        self.losses.append(loss.item())


def main() -> None:
    """Run the paper's loop with the ACT agent, then optionally evaluate."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="move_plate")
    parser.add_argument("--demos", type=int, default=60)
    parser.add_argument(
        "--steps", type=int, default=100_000, help="env steps (one gradient step each)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="run the evaluation protocol at the end",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)

    env = make_gym(args.task)
    agent = Agent(env.observation_space, env.action_space)
    agent.ingest(env.get_demos(args.demos))

    obs, _ = env.reset(seed=args.seed)
    episodes, successes, t0 = 0, 0, time.time()
    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        nxt, reward, term, trunc, info = env.step(action)
        agent.observe(obs, action, reward, nxt, term, trunc)
        agent.update()
        obs = nxt
        if term or trunc:
            episodes += 1
            successes += int(info.get("success", reward > 0))
            obs, _ = env.reset()
        if step % 200 == 0:
            recent = (
                float(np.mean(agent.losses[-200:])) if agent.losses else float("nan")
            )
            print(
                f"step {step:6d} | L1 {recent:.4f} | episodes {episodes} | "
                f"successes {successes} | {step / (time.time() - t0):.1f} steps/s"
            )
    env.close()

    if args.eval_episodes > 0:
        from bigym.loco.eval import evaluate

        # The protocol runner drives a fresh official make() env and hands the
        # policy dm_env-style TimeSteps; adapt them to the gymnasium dict.
        class Policy:
            """dm_env TimeStep -> gymnasium dict adapter for the protocol runner."""

            def __call__(self, timestep):
                return agent.act(
                    {"rgb": timestep.rgb_obs, "state": timestep.low_dim_obs}
                )

            def reset(self):
                agent.ensemble.clear()

        result = evaluate(
            Policy(),
            task_name=args.task,
            method="act-example",
            episodes=args.eval_episodes,
        )
        # A full 100-episode protocol run also carries result["record"], the
        # leaderboard entry stamped with the substrate fingerprint.
        print(
            f"evaluation: success_rate={result['success_rate']:.3f} over "
            f"{result['episodes']} episodes"
            + (" (leaderboard record produced)" if "record" in result else "")
        )


if __name__ == "__main__":
    main()
