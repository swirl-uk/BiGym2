r"""Train an ACT-style policy on BiGym demonstrations, then evaluate it.

ACT learns from the demonstrations alone, so it trains offline and touches the
env only to be evaluated::

    from bigym.loco import make_gym

    env = make_gym("reach_target_single")
    demos = env.get_demos(60, only_successful=True, decode_images=False)
    # ... behaviour cloning on demos ...
    bigym.loco.eval.evaluate(policy, task_name="reach_target_single", ...)

The README's gymnasium loop (``observe`` and ``update`` while acting) is the
shape of an online method that keeps learning from its own interaction; an
imitation method like this one has no use for it.

The policy is a compact Action Chunking Transformer: a CNN per camera and a
state MLP feed a transformer decoder that predicts the next ``chunk`` actions,
trained with an L1 loss and executed with temporal ensembling. It leaves out
parts of the original ACT (the CVAE, a pretrained ResNet backbone). This
simplified ACT example is for getting started; it does not reproduce the
paper's ACT results::

    MUJOCO_GL=egl uv run --with torch --with imageio-ffmpeg python examples/train_act.py \
        --task reach_target_single --record 3

saves the policy to ``act_runs/<task>/policy.pt``, scores it on the 100
evaluation seeds and records three evaluation episodes into the same folder,
each as an mp4 and as the simulator states the 3D viewer plays::

    uv run bigym-view --demo-dir act_runs/reach_target_single

``--out`` picks another folder, which must be empty.
``--load act_runs/<task>/policy.pt`` evaluates or records a saved policy
without training it again.

``decode_images=False`` keeps the demonstration frames compressed in memory;
the data loader workers decode only the frames each batch samples. torch and
imageio-ffmpeg (for the mp4s) are all this needs beyond the core install:
``--with`` adds them for this run without touching the project environment.
On macOS drop ``MUJOCO_GL=egl``.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from bigym.loco import make, make_gym
from bigym.loco.demos.schema import BATCH_FORMAT
from bigym.loco.eval import eval_seeds, evaluate, is_success


class DemoChunks(Dataset):
    """Every demonstration frame with the chunk of actions that follows it.

    The chunk is padded with the episode's last action.
    """

    def __init__(self, demos: list[dict], chunk: int):
        """Index the frames of ``demos`` (``get_demos`` trajectories)."""
        self.demos = demos
        self.chunk = chunk
        self.frames = [
            (d, t) for d, demo in enumerate(demos) for t in range(len(demo["action"]))
        ]

    def __len__(self) -> int:
        """Number of training frames."""
        return len(self.frames)

    def __getitem__(self, index: int):
        """``(rgb, state, actions)`` of one frame."""
        d, t = self.frames[index]
        demo = self.demos[d]
        actions = demo["action"][t : t + self.chunk]
        if len(actions) < self.chunk:
            pad = np.repeat(actions[-1:], self.chunk - len(actions), axis=0)
            actions = np.concatenate([actions, pad])
        return demo["obs"]["rgb"][t], demo["obs"]["state"][t], actions


def random_shift(rgb: torch.Tensor, pad: int) -> torch.Tensor:
    """Shift every image by up to ``pad`` pixels, filling with the edge."""
    b, c, h, w = rgb.shape
    padded = F.pad(rgb, (pad, pad, pad, pad), mode="replicate")
    dx, dy = torch.randint(0, 2 * pad + 1, (2, b), device=rgb.device)
    rows = torch.arange(h, device=rgb.device)[None, :] + dy[:, None]
    cols = torch.arange(w, device=rgb.device)[None, :] + dx[:, None]
    index = torch.arange(b, device=rgb.device)[:, None, None]
    return padded.permute(0, 2, 3, 1)[
        index, rows[:, :, None], cols[:, None, :]
    ].permute(0, 3, 1, 2)


class ACT(nn.Module):
    """CNN per camera + state MLP -> transformer decoder -> a chunk of actions."""

    def __init__(self, rgb_shape, state_mean, state_std, action_dim, chunk, hidden=256):
        """Build the network for ``(cams, channels, H, W)`` images."""
        super().__init__()
        cams, channels = rgb_shape[0], rgb_shape[1]
        self.register_buffer("state_mean", torch.as_tensor(state_mean))
        self.register_buffer("state_std", torch.as_tensor(state_std))
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
            nn.Linear(len(state_mean), hidden), nn.LayerNorm(hidden)
        )
        self.cam_embed = nn.Parameter(torch.zeros(cams, hidden))
        layer = nn.TransformerDecoderLayer(
            hidden, 4, hidden * 2, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.queries = nn.Parameter(torch.randn(chunk, hidden) * 0.02)
        self.head = nn.Linear(hidden, action_dim)

    def forward(self, rgb, state, shift=0):
        """``(B, cams, C, H, W)`` uint8 and ``(B, D)`` -> ``(B, chunk, A)`` in [-1, 1]."""
        b, cams = rgb.shape[:2]
        x = rgb.flatten(0, 1).float() / 255.0
        if shift:
            x = random_shift(x, shift)
        tokens = self.cnn(x).view(b, cams, -1) + self.cam_embed
        state = (state - self.state_mean) / self.state_std
        memory = torch.cat([tokens, self.state(state)[:, None]], dim=1)
        queries = self.queries[None].expand(b, -1, -1)
        return torch.tanh(self.head(self.decoder(queries, memory)))


class EnsemblePolicy:
    """Evaluation-time policy: temporal ensembling over the predicted chunks."""

    def __init__(self, model: ACT, device, chunk: int, gain: float = 0.01):
        """Wrap a trained model."""
        self.model, self.device, self.chunk, self.gain = model, device, chunk, gain
        self.chunks: collections.deque = collections.deque(maxlen=chunk)

    def reset(self):
        """Forget the chunks of the previous episode."""
        self.chunks.clear()

    def __call__(self, timestep) -> np.ndarray:
        """The action for one evaluation ``TimeStep``."""
        with torch.no_grad():
            rgb = torch.as_tensor(np.asarray(timestep.rgb_obs), device=self.device)
            state = torch.as_tensor(
                np.asarray(timestep.low_dim_obs),
                dtype=torch.float32,
                device=self.device,
            )
            self.chunks.append(self.model(rgb[None], state[None])[0].cpu().numpy())
        # Chunk k steps old predicted this step as its k-th action; older
        # chunks weigh more, as in ACT.
        preds = np.stack([c[k] for k, c in enumerate(reversed(self.chunks))])
        weights = np.exp(-self.gain * np.arange(len(preds)))[::-1]
        action = np.average(preds, axis=0, weights=weights)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def train(args) -> tuple[ACT, dict]:
    """Train on the task's demonstrations; return the model and its shapes."""
    env = make_gym(args.task)
    demos = env.get_demos(args.demos, only_successful=True, decode_images=False)
    env.close()
    data = DemoChunks(demos, args.chunk)
    states = np.concatenate([demo["obs"]["state"] for demo in demos])
    shapes = {
        "rgb_shape": list(demos[0]["obs"]["rgb"].shape[1:]),
        "state_dim": states.shape[1],
        "action_dim": demos[0]["action"].shape[1],
    }
    model = ACT(
        shapes["rgb_shape"],
        states.mean(0).astype(np.float32),
        states.std(0).astype(np.float32) + 1e-3,
        shapes["action_dim"],
        args.chunk,
    ).to(args.device)
    print(f"{len(demos)} demos, {len(data)} training frames")

    loader = DataLoader(
        data,
        batch_size=args.batch,
        sampler=RandomSampler(
            data, replacement=True, num_samples=args.steps * args.batch
        ),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=args.device.type == "cuda",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    losses, t0 = [], time.time()
    for step, (rgb, state, actions) in enumerate(loader, start=1):
        rgb, state, actions = (
            x.to(args.device, non_blocking=True) for x in (rgb, state, actions)
        )
        loss = F.l1_loss(model(rgb, state, shift=args.shift), actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(
                f"step {step:6d} | L1 {np.mean(losses[-500:]):.4f} | "
                f"{step / (time.time() - t0):.1f} steps/s"
            )
    return model, shapes


def load(path: Path, device) -> tuple[ACT, dict]:
    """A policy saved by this script, and what it was saved with."""
    saved = torch.load(path, map_location=device)
    state_dim = saved["state_dim"]
    model = ACT(
        saved["rgb_shape"],
        np.zeros(state_dim, np.float32),
        np.ones(state_dim, np.float32),
        saved["action_dim"],
        saved["chunk"],
    ).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


def record(policy: EnsemblePolicy, task: str, episodes: int, out: Path) -> None:
    """Run the policy on the first evaluation seeds and keep what it did.

    Episode ``seed<S>`` becomes ``seed<S>.mp4``, seen from outside the robot,
    and ``seed<S>.npz``, the simulator state of every step, which
    ``bigym-view`` plays with the ``metadata.json`` next to it.
    """
    import imageio

    env = make(task)
    metadata = {
        "format": BATCH_FORMAT,
        "control_step_seconds": env.control_step_seconds,
    }
    metadata.update(env.config.to_metadata(task))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=1))
    data = env.inner_env.data
    for seed in eval_seeds(episodes):
        policy.reset()
        timestep = env.reset(seed=seed)
        qpos, qvel, rewards = [data.qpos.copy()], [data.qvel.copy()], []
        with imageio.get_writer(out / f"seed{seed}.mp4", fps=25) as video:
            while not timestep.last():
                timestep = env.step(policy(timestep))
                qpos.append(data.qpos.copy())
                qvel.append(data.qvel.copy())
                rewards.append(timestep.reward)
                if len(rewards) % 2:  # 50 Hz control, 25 fps video
                    video.append_data(env.render())  # ty: ignore[unresolved-attribute]
        success = is_success(env)
        np.savez(
            out / f"seed{seed}.npz",
            full_qpos=np.array(qpos),
            full_qvel=np.array(qvel),
            reward=np.array(rewards, dtype=np.float32).reshape(-1, 1),
            seed=np.int64(seed),
            success=np.float32(success),
            length=np.int64(len(rewards)),
        )
        print(f"recorded seed {seed}: {'success' if success else 'failure'}")
    env.close()


def main() -> None:
    """Train (or load) a policy, evaluate it, record some episodes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="reach_target_single")
    parser.add_argument("--demos", type=int, default=60)
    parser.add_argument("--steps", type=int, default=20_000, help="gradient steps")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--shift", type=int, default=4, help="augmentation, pixels")
    parser.add_argument("--workers", type=int, default=4, help="data loader workers")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--record", type=int, default=0, help="episodes to record")
    parser.add_argument("--load", type=Path, help="a saved policy.pt; skips training")
    parser.add_argument("--out", type=Path, help="default: act_runs/<task>")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    home = args.load.parent if args.load else Path("act_runs") / args.task
    out = args.out or home
    # A folder holds one policy and its recordings.
    reuse = args.load is not None and out.resolve() == home.resolve()
    if not reuse and out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty; pass another --out")
    if args.load:
        model, saved = load(args.load, args.device)
        args.task, args.chunk = saved["task"], saved["chunk"]
        out.mkdir(parents=True, exist_ok=True)
    else:
        model, shapes = train(args)
        out.mkdir(parents=True, exist_ok=True)
        saved = {"task": args.task, "chunk": args.chunk, **shapes}
        torch.save({**saved, "model": model.state_dict()}, out / "policy.pt")
        print(f"saved {out / 'policy.pt'}")

    model.eval()
    policy = EnsemblePolicy(model, args.device, args.chunk)
    if args.eval_episodes > 0:
        result = evaluate(
            policy,
            task_name=args.task,
            method="act-example",
            episodes=args.eval_episodes,
        )
        # A full 100-episode run also carries result["record"], the
        # leaderboard entry stamped with the substrate fingerprint.
        print(
            f"evaluation: success_rate={result['success_rate']:.3f} over "
            f"{result['episodes']} episodes"
        )
    if args.record > 0:
        record(policy, args.task, args.record, out)
        print(f"watch in 3D: uv run bigym-view --demo-dir {out}")


if __name__ == "__main__":
    main()
