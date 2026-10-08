"""Shared onnxruntime session policy for the lower-body adapters.

Inference here is ~20 us per call, but ORT's default options spin an
intra-op pool sized to the machine's cores that busy-waits between calls
(~2 idle cores: 300% -> 100% process CPU at equal throughput, bit-identical
outputs). The vendored GR00T runtime creates its
sessions with these options so the policy lives in exactly one place.
"""

from __future__ import annotations

import onnxruntime as ort


def single_thread_session_options():
    """Return ORT SessionOptions pinned to one intra-op/inter-op thread."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return opts
