# The agent container's network

An agent session gets one network capability: its own model endpoint. Every
other host — package indexes, code hosts, search engines, the dataset on the
Hub — is refused, because the environment is open source and a fetch or a
search would be a ground-truth channel into a benchmark that is supposed to
measure control design.

That is arranged with two docker networks and one proxy:

- `bigym-agent-internal` is an **internal** network (`--internal`): containers
  on it have no route off the host. The agent container joins only this one.
- `bigym-agent-egress` is an ordinary bridge with egress. Only the proxy joins
  it.
- `bigym-agent-proxy` is a [tinyproxy](https://tinyproxy.github.io/) on both
  networks. It accepts HTTP/CONNECT from the internal side and forwards only
  the hosts in its filter file (`FilterDefaultDeny Yes`), so the allowlist is
  the whole of the session's reachable internet.

`bigym-agent run` passes `HTTPS_PROXY` / `HTTP_PROXY` (`--proxy`, default
`http://bigym-agent-proxy:8888`) and joins the container to `--docker-network`
(default `bigym-agent-internal`).

## Set it up

Nothing to do by hand: `bigym-agent run` creates the two networks, starts
the proxy with the shipped `proxy/filter` allowlist (the model endpoints of
every shipped harness) and builds the harness image from its Dockerfile, each
only when it is missing. `bigym-agent proxy status` shows what is there,
`proxy up --harness codex` prepares a host ahead of time, `proxy down`
removes the proxy container. Pass `--no-docker-setup` to `run` to create
nothing (for instance with a `--docker-network` / `--proxy` of your own; a
non-default value is never touched).

The equivalent manual commands, should you want to run the proxy elsewhere:

```bash
cd bigym/loco/agent/docker            # the directory holding proxy/ and the Dockerfiles

docker network create --internal bigym-agent-internal
docker network create bigym-agent-egress

docker run -d --name bigym-agent-proxy --restart unless-stopped \
  --network bigym-agent-egress \
  -v "$PWD/proxy/tinyproxy.conf:/etc/tinyproxy/tinyproxy.conf:ro" \
  -v "$PWD/proxy/filter:/etc/tinyproxy/filter:ro" \
  vimagick/tinyproxy
docker network connect bigym-agent-internal bigym-agent-proxy
```

The filter file is read when the proxy starts; edit it and recreate the
container to change the allowlist.

## Build the agent images

The default image names are `bigym-agent-codex` and `bigym-agent-claude`;
`bigym-agent run --container IMAGE` overrides them.

```bash
cd bigym/loco/agent/docker
docker build -f Dockerfile.codex  -t bigym-agent-codex  .
docker build -f Dockerfile.claude -t bigym-agent-claude .
```

The Codex and Claude Code images install the latest CLI; `--build-arg` pins a
version (`CODEX_VERSION`, `CLAUDE_CODE_VERSION`).
The images carry python with the libraries a submitted policy may use, ffmpeg
and ripgrep, and no pip, curl or git — the sandbox's `./python` wrapper points
at that interpreter.

## Check it

From inside the network, the model endpoint answers and everything else does
not:

```bash
docker run --rm --network bigym-agent-internal \
  -e HTTPS_PROXY=http://bigym-agent-proxy:8888 bigym-agent-codex \
  node -e "fetch('https://api.openai.com/v1/models').then(r=>console.log(r.status))"
```

A refused host returns a tinyproxy error page (HTTP 403), not a connection
error; `docker logs bigym-agent-proxy` shows each refusal with the host that
was asked for, which is also how you find a host the row legitimately needs.

## Without docker

`--isolation soft` runs the harness on the host with its own tool rules
instead (`bigym.loco.agent.settings`). Nothing then enforces the allowlist:
audit the session afterwards with `bigym-agent audit <root>`, which flags
every network command in the transcript.
