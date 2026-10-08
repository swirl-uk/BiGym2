"""Transcript adapters: each harness's event stream, the totals and the markdown.

The fixtures below are synthetic lines in the shapes each harness emits, so
the adapters are tested against the real event formats.
"""

from __future__ import annotations

import json

import pytest

from bigym.loco.agent import transcript as tr

# `codex exec --json`: a thread id, a message, a command with its output, an
# edit, and the cumulative usage counter of the finished turn.
CODEX_LINES = [
    {"type": "thread.started", "thread_id": "00000000-0000-4000-8000-000000000001"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {
            "id": "item_0",
            "type": "agent_message",
            "text": "I'll read the control API, then build a policy.\n",
        },
    },
    {
        "type": "item.started",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/sh -lc 'ls docs'",
            "aggregated_output": "",
            "exit_code": None,
            "status": "in_progress",
        },
    },
    {
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/sh -lc 'ls docs'",
            "aggregated_output": "api.md\n",
            "exit_code": 0,
            "status": "completed",
        },
    },
    {
        "type": "item.completed",
        "item": {
            "id": "item_54",
            "type": "file_change",
            "changes": [{"path": "/work/policy.py", "kind": "update"}],
            "status": "completed",
        },
    },
    {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 5554056,
            "cached_input_tokens": 5442432,
            "cache_write_input_tokens": 0,
            "output_tokens": 33750,
            "reasoning_output_tokens": 18665,
        },
    },
]

# `claude -p --output-format stream-json --verbose`.
CLAUDE_LINES = [
    {
        "type": "system",
        "subtype": "init",
        "cwd": "/work",
        "session_id": "00000000-0000-4000-8000-000000000002",
        "model": "test-model",
        "permissionMode": "bypassPermissions",
    },
    {
        "type": "assistant",
        "message": {
            "model": "test-model",
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Read the docs first."}],
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 15185,
                "cache_read_input_tokens": 0,
                "output_tokens": 3,
            },
        },
        "session_id": "00000000-0000-4000-8000-000000000002",
        "timestamp": "2026-09-15T15:24:52.000Z",
    },
    {
        "type": "assistant",
        "message": {
            "model": "test-model",
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "Bash",
                    "input": {"command": "ls -la", "description": "List files"},
                }
            ],
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 15185,
                "cache_read_input_tokens": 0,
                "output_tokens": 3,
            },
        },
        "timestamp": "2026-09-15T15:24:53.000Z",
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "tool_use_id": "toolu_01",
                    "type": "tool_result",
                    "content": "PROMPT.md\npolicy.py\n",
                }
            ],
        },
        "timestamp": "2026-09-15T15:24:54.000Z",
    },
    {
        "type": "system",
        "subtype": "permission_denied",
        "tool_name": "Bash",
        "message": "a command the shell parser cannot analyze asks the person",
    },
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 120,
        "duration_ms": 1800000,
        "total_cost_usd": 12.5,
        "usage": {
            "input_tokens": 5478,
            "cache_creation_input_tokens": 271405,
            "cache_read_input_tokens": 25039120,
            "output_tokens": 154148,
            "output_tokens_details": {"thinking_tokens": 90000},
        },
    },
]


def _write(path, lines):
    """Write fixture events as JSON lines."""
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_codex_adapter(tmp_path):
    """Codex events become messages, commands with output, edits and usage."""
    records = list(tr.from_codex_events(_write(tmp_path / "e.jsonl", CODEX_LINES)))
    kinds = [r["kind"] for r in records]
    assert kinds == ["meta", "message", "command", "output", "edit", "usage"]
    command = records[2]
    assert command["command"] == "/bin/sh -lc 'ls docs'"
    assert command["exit_code"] == 0
    assert records[3]["text"] == "api.md\n"
    assert records[4]["path"] == "/work/policy.py"
    tokens = records[5]["tokens"]
    # The counter is cumulative; the first record reports the whole of it, with
    # the cached tokens split out of the input count.
    assert tokens == {
        "input": 5554056 - 5442432,
        "cached": 5442432,
        "output": 33750,
        "reasoning": 18665,
    }


def test_codex_usage_is_additive(tmp_path):
    """A second turn reports only its increment, so records sum to the total."""
    lines = CODEX_LINES + [
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 6000000,
                "cached_input_tokens": 5800000,
                "output_tokens": 40000,
                "reasoning_output_tokens": 20000,
            },
        }
    ]
    records = list(tr.from_codex_events(_write(tmp_path / "e.jsonl", lines)))
    total = tr.usage_summary(records)["tokens"]
    assert total == {
        "input": 6000000 - 5800000,
        "cached": 5800000,
        "output": 40000,
        "reasoning": 20000,
    }


def test_claude_adapter(tmp_path):
    """Claude stream-json becomes reasoning, commands, output, meta and usage."""
    records = list(tr.from_claude_stream(_write(tmp_path / "s.jsonl", CLAUDE_LINES)))
    kinds = [r["kind"] for r in records]
    assert kinds == [
        "meta",
        "reasoning",
        "command",
        "output",
        "meta",
        "meta",
        "usage",
    ]
    assert records[1]["text"] == "Read the docs first."
    assert records[2]["command"] == "ls -la"
    assert records[2]["t"] == pytest.approx(tr.iso_to_unix("2026-09-15T15:24:53.000Z"))
    assert "permission denied" in records[4]["text"]
    usage = records[-1]
    assert usage["usd"] == 12.5
    assert usage["tokens"]["cached"] == 25039120
    assert usage["tokens"]["input"] == 5478 + 271405
    assert usage["tokens"]["reasoning"] == 90000
    # A harness that reports its own cost is believed over the price table.
    summary = tr.usage_summary(records, "test-model")
    assert summary["cost_source"] == "harness"
    assert summary["usd"] == pytest.approx(12.5)


def test_claude_usage_falls_back_when_cut_short(tmp_path):
    """Without a result event the per-message blocks are summed, once each."""
    records = list(
        tr.from_claude_stream(_write(tmp_path / "s.jsonl", CLAUDE_LINES[:-1]))
    )
    usage = [r for r in records if r["kind"] == "usage"]
    assert len(usage) == 1
    # Both assistant events belong to one message: its usage counts once.
    assert usage[0]["tokens"]["input"] == 2 + 15185


def test_price_table():
    """Model prices are looked up exactly, then by prefix."""
    sol = tr.price_of("gpt-6-sol")
    assert sol is not None and sol["output"] == 10.0
    astra = tr.price_of("gpt-6-astra-2026-09-01")
    assert astra is not None and astra["input"] == 10.0
    assert tr.price_of("a-model-nobody-priced") is None
    assert tr.price_of(None) is None


def test_usage_summary_prices_from_the_table(tmp_path):
    """A harness that reports no cost is priced from MODEL_PRICES."""
    records = list(tr.from_codex_events(_write(tmp_path / "e.jsonl", CODEX_LINES)))
    summary = tr.usage_summary(records, "gpt-6-astra")
    expected = ((5554056 - 5442432) * 10.0 + 5442432 * 1.0 + 33750 * 50.0) / 1e6
    assert summary["usd"] == pytest.approx(round(expected, 4))
    assert summary["cost_source"] == "price_table"
    assert summary["pricing_date"] == tr.PRICING_DATE
    assert tr.usage_summary(records, None)["cost_source"] == "unknown"


def test_markdown_renders_commands_with_their_output(tmp_path):
    """Commands are fenced blocks carrying the following output record."""
    records = list(tr.from_codex_events(_write(tmp_path / "e.jsonl", CODEX_LINES)))
    text = tr.render_markdown(records, "move_plate (codex)")
    assert text.startswith("# move_plate (codex)")
    assert "$ /bin/sh -lc 'ls docs'" in text
    # The output is inside the command's fence, not in a second block.
    assert text.count("```console") == 1
    assert "api.md" in text
    assert "edit `/work/policy.py`" in text


def test_markdown_truncates_long_output():
    """An output block is quoted up to MD_OUTPUT_LINES lines."""
    records = [
        {"kind": "command", "role": "agent", "command": "seq 500", "exit_code": 0},
        {
            "kind": "output",
            "role": "harness",
            "text": "\n".join(str(i) for i in range(500)),
        },
    ]
    text = tr.render_markdown(records, "t")
    assert f"[{500 - tr.MD_OUTPUT_LINES} more lines]" in text
    assert "\n499\n" not in text


def test_write_transcript_and_report(tmp_path, capsys):
    """write_transcript reads raw/, writes both files, and `report` redoes it."""
    cell = tmp_path / "move_plate"
    (cell / "raw").mkdir(parents=True)
    _write(cell / "raw" / "codex_events.jsonl", CODEX_LINES)
    (cell / "run.json").write_text(json.dumps({"model": "gpt-6-astra"}))
    summary = tr.write_transcript(cell, tr.cell_model(cell))
    assert summary["harness"] == "codex"
    assert summary["records"] == 6
    assert summary["cost_source"] == "price_table"
    lines = (cell / "transcript.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines][0] == "meta"
    assert (cell / "transcript.md").read_text().startswith("# move_plate (codex)")
    assert tr.main([str(cell)]) == 0
    assert "codex, 6 records" in capsys.readouterr().out


def test_custom_adapter(tmp_path):
    """A custom agent's stdout becomes one output record and no usage at all."""
    cell = tmp_path / "reach_target_single"
    (cell / "raw").mkdir(parents=True)
    stdout = cell / "raw" / "custom_stdout.txt"
    stdout.write_text("wrote policy.py\nsuccess 2/2\n")
    records = list(tr.from_custom(stdout))
    assert [r["kind"] for r in records] == ["output"]
    assert records[0]["role"] == "harness"
    assert "success 2/2" in records[0]["text"]
    assert isinstance(records[0]["t"], float)
    # the cell directory is accepted as well as the file itself
    assert list(tr.from_custom(cell)) == records
    # the cell's raw stream is found without being named
    assert tr.find_raw(cell) == ("custom", stdout)
    summary = tr.write_transcript(cell, None)
    assert summary["harness"] == "custom"
    assert summary["records"] == 1
    # nothing reports tokens for a command we know nothing about
    assert summary["tokens"] == {
        "input": 0,
        "cached": 0,
        "output": 0,
        "reasoning": 0,
    }
    assert summary["usd"] is None
    assert summary["cost_source"] == "unknown"
    text = (cell / "transcript.md").read_text()
    assert text.startswith("# reach_target_single (custom)")
    assert "success 2/2" in text


def test_custom_adapter_ignores_an_empty_stream(tmp_path):
    """A command that printed nothing leaves no record to render."""
    cell = tmp_path / "t"
    (cell / "raw").mkdir(parents=True)
    (cell / "raw" / "custom_stdout.txt").write_text("   \n")
    assert list(tr.from_custom(cell)) == []
    assert list(tr.from_custom(cell / "raw" / "nothing.txt")) == []


def test_write_transcript_without_a_stream(tmp_path):
    """A cell with no raw stream writes nothing and says so."""
    cell = tmp_path / "empty"
    cell.mkdir()
    assert tr.write_transcript(cell)["harness"] is None
    assert not (cell / "transcript.md").exists()
    assert tr.main([str(cell)]) == 1


def test_audit_flags_what_left_the_sandbox(tmp_path):
    """Host paths, the cell's own files and network commands are flagged."""
    from bigym.loco.agent import audit

    cell = tmp_path / "root" / "move_plate"
    (cell / "sandbox").mkdir(parents=True)
    records = [
        {"kind": "command", "role": "agent", "command": "ls /work/docs"},
        {"kind": "command", "role": "agent", "command": f"cat {cell}/ledger.jsonl"},
        {"kind": "command", "role": "agent", "command": "curl -s https://example.com"},
        {
            "kind": "command",
            "role": "agent",
            "command": "./python -c 'nc=len(q); f(nc)'",
        },
        {"kind": "edit", "role": "agent", "path": "/work/policy.py"},
    ]
    (cell / "transcript.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    calls, flagged = audit.audit_cell(cell)
    assert calls == 5
    assert len(flagged) == 2
    assert any("ledger.jsonl" in f for f in flagged)
    assert any("curl" in f for f in flagged)
    # the sandbox, mounted at /work, is rewritten before the test
    assert audit.find_cells(tmp_path / "root") == [cell]
    assert audit.main([str(tmp_path / "root")]) == 1


def test_usage_row(tmp_path):
    """A cell's usage row joins run.json's configuration to the transcript."""
    from bigym.loco.agent import usage

    cell = tmp_path / "root" / "move_plate"
    (cell / "raw").mkdir(parents=True)
    _write(cell / "raw" / "codex_events.jsonl", CODEX_LINES)
    (cell / "run.json").write_text(
        json.dumps(
            {
                "harness": "codex",
                "model": "gpt-6-astra",
                "effort": "high",
                "interface": "strict",
                "verdict": {"state": "ok", "reason": ""},
                "wall_clock_s": 12.5,
            }
        )
    )
    row = usage.session_row(cell)
    assert row is not None
    assert row["task"] == "move_plate"
    assert row["verdict"] == "ok"
    assert row["cached_tokens"] == 5442432
    assert row["cost_source"] == "price_table"
    assert set(usage.COLUMNS) == set(row)
    out = tmp_path / "usage.csv"
    assert usage.main([str(tmp_path / "root"), "--csv", str(out)]) == 0
    assert "move_plate" in out.read_text()
    assert usage.session_row(tmp_path) is None
