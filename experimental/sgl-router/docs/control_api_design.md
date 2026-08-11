# Design — sgl-router control / profiling API

> **v1 shipped in this branch — minimal per-worker admin, no `/control` prefix.**
> Implemented endpoints (each proxies to the addressed worker's engine):
>
> | Method | Path | Engine call |
> |---|---|---|
> | POST | `/workers/{id}/profiling/start` | `POST {url}/start_profile` (body forwarded) |
> | POST | `/workers/{id}/profiling/end` | `POST {url}/stop_profile` |
> | POST | `/workers/{id}/cache/clean` | `POST {url}/flush_cache` |
>
> `{id}` is the registry worker id; `404` unknown id, `200` on 2xx upstream,
> `502` otherwise. Torch traces are written on the engine host (its `output_dir`
> / `SGLANG_TORCH_PROFILER_DIR`) — the router only triggers and reports; it does
> not return trace files (put `output_dir` on a shared mount to collect them).
> Code: `server/routes/workers_admin.rs`.
>
> **The rest of this document (§2–§9) is the broader future design** — the
> `/control/*` plane, stable logical ids, diagnostics, log-level, mori-trace
> compat — kept for reference; not yet implemented.

## 1. Summary

`sgl-router` today exposes a rich data plane (`/v1/chat/completions`,
`/v1/tokenize`, …), health probes (`/healthz`, `/readyz`), Prometheus
(`/metrics`), and exactly **one** control operation (`POST /flush_cache`). It has
no way to, at runtime: toggle profiling, dump internal stats, change log level,
inspect cluster/worker state, or drive per-worker GPU profiling.

mori-scheduler has all of this (`/profile/*`, `/router/config/log-level`,
`/cluster/status`, `/workers/:id/*`, and the standalone `mori-trace` collector).

This design adds a **`/control/*` control plane** to sgl-router that:

- is **off by default** and **guarded** (opt-in flag + bearer token / bind
  scope) — control endpoints mutate runtime behavior;
- is **pluggable / self-registering** — a module contributes control endpoints
  only when its feature is active (mirroring the `MetricCollector` pattern), so
  the control surface reflects exactly what is running;
- **proxies to the SGLang engines** for per-worker actions the engine already
  supports (`/start_profile`, `/stop_profile`, `/flush_cache`), reusing the
  existing `/flush_cache` fan-out shape;
- stays **measurement/ops-only** — it does not add scheduling or placement logic.

## 2. Gap vs mori-scheduler

| Capability | mori-sched | sgl-router today | this design |
|---|---|---|---|
| Fleet cache flush | `/cache/flush` | `POST /flush_cache` | `GET /control/cache`, `POST /control/cache/clean` + per-worker |
| Router profiling toggle | `POST /profile/control` | — | `PUT /control/profiling {enabled}` |
| Profiling stats / dump | `/profile/stats`, `/profile/dump` | — | `GET /control/profiling/{stats,records}` |
| Per-worker GPU/torch profile | `/workers/:id/profile/{start,stop}` | — | proxy → engine `/start_profile` `/stop_profile` |
| Runtime log level | `GET/PUT /router/config/log-level` | — | `GET/PUT /control/config/log-level` |
| Config reload | `POST /router/config/reload` | — | `POST /control/config:reload` (P3, best-effort) |
| Cluster status | `/cluster/status`, `/workers` | (partly via `/metrics`) | `GET /control/status`, `/control/workers` |
| Trace collector | `mori-trace` binary | — | reuse `mori-trace` (it just scrapes `/metrics`); optional `sgl-trace` later |

## 3. Design principles

1. **Separate plane, single prefix.** All control endpoints live under
   `/control/*`, never mixed with the data plane. `/metrics`, `/healthz`,
   `/readyz` stay where they are (scrapers/orchestrators depend on them).
2. **Off by default + guarded.** The control plane is built only when
   `--enable-control-api` (or `SGL_ROUTER_CONTROL`) is set. When on, every
   `/control/*` request must present `Authorization: Bearer <token>` where the
   token is `SGL_ROUTER_CONTROL_TOKEN`. Optionally bind the control plane to a
   separate address (`--control-addr 127.0.0.1:30001`) so it isn't exposed on
   the data port at all. Mutating endpoints are POST/PUT only.
3. **Pluggable / self-registering** (same ethos as `MetricCollector`). A
   `ControlEndpoint` provider registers its routes only when its module is
   active. The router's control surface = the union of what active modules
   registered + the always-on core (status, log-level, cache).
4. **Proxy, don't reinvent.** Per-worker actions (GPU profile, flush) fan out to
   the engines' existing endpoints via the proxy client, reusing
   `cache::fan_out_flush`'s pattern (breaker-bypass, per-worker result).
5. **Measurement/ops-only.** No routing/placement/eviction decisions here.
6. **Clients address logical workers, not physical endpoints.** Everything a
   client lists or operates on is a *logical worker* keyed by a **stable id**
   (see §3.1). The router maps that id to whatever physical endpoint currently
   backs it.

## 3.1 Logical worker identity (stable ids)

Clients only ever see and operate on **logical workers**. Each has an `id` that
must be **stable across restarts** so an operator can script against
`/control/workers/{id}/...` and keep hitting the same logical slot even after the
backing process/pod is replaced.

**Why a new layer is needed** — today's ids are *physical* and not restart-stable:
- static discovery uses the **raw URL** as the id (`static_urls.rs:41`,
  `WorkerId(url)`) — changes if the IP/host changes, and is an awkward client key;
- k8s uses the **pod UID** (`k8s.rs:123`, `{ns}/{uid}`) — a fresh UID every pod
  incarnation, i.e. it **changes on every restart** (intentionally, to reset the
  breaker). Good for internal lifecycle, unusable as a stable client handle.

So we separate two identities:
- **Physical id** (`WorkerId`, unchanged): URL / pod-UID. Drives registry
  lifecycle, circuit breaker, discovery Add/Remove. May change across restarts.
- **Logical id** (new, client-facing): a stable slug. The control API's `{id}`
  is always this. The router keeps a `logical_id ↔ WorkerId` map and resolves it
  per request; when a pod restarts with a new UID/IP but the same logical id, the
  client keeps using the same `{id}`.

**Logical id resolution (first match wins):**
1. **Operator-assigned** — most durable, survives IP/UID changes:
   - static: `--worker-urls` accepts `id=<url>` (e.g. `prefill-0=http://10.0.0.1:31100`);
   - k8s: pod label/annotation `sglang.ai/worker-id`.
2. **k8s StatefulSet pod name** (e.g. `prefill-0`) from `target_ref.name` —
   stable across restarts (unlike the UID). Preferred default on k8s.
3. **Deterministic derivation** — `"{role}-{ordinal}"` where `ordinal` is the
   rank of the worker within its role after a **stable sort of the configured
   URLs** (host:port). Stable as long as the URL set is unchanged. Fallback:
   the normalized URL.

This keeps existing internal behavior (breaker/lifecycle keyed by physical
`WorkerId`) while giving clients a durable handle. `GET /control/workers` returns
both so the mapping is observable: `{id (logical), url (physical), mode, …}`.

## 4. RESTful API specification

Resource-oriented, not RPC. Nouns for resources; verbs carry the intent:
**GET** read, **PUT** idempotent set-state, **DELETE** remove/clear, **POST**
only for non-idempotent actions that don't map to a resource verb.

### 4.0 Conventions
- Base path `/control`; all endpoints require `Authorization: Bearer <token>`.
- Request/response bodies are JSON (`application/json`).
- Error body: `{ "error": "<code>", "detail": "<human message>" }`.
- Status codes: `200` OK · `202` Accepted (async/fan-out kicked off) · `204` No
  Content · `400` bad body · `401` unauthorized · `404` unknown resource ·
  `409` conflict (illegal state transition) · `502` upstream worker error ·
  `503` not ready.
- Fleet (multi-worker) operations return per-worker results and use `200` when
  all succeeded, `502` when any worker failed (with the per-worker breakdown in
  the body) — mirrors today's `/flush_cache`.
- `{id}` is the **stable logical worker id** (§3.1), never the raw URL/UID. An
  unknown id → `404`.

### 4.1 Cluster (read-only)
| Verb | Resource | → |
|---|---|---|
| `GET` | `/control/status` | `200` `{version, uptime_s, ready, models:[{id,policy}], workers:{prefill,decode,plain,total}}` |

### 4.2 Workers (read-only collection)
| Verb | Resource | → |
|---|---|---|
| `GET` | `/control/workers` | `200` `[{id,url,mode,healthy,cb_state,inflight,model_ids}]` — `id` = stable logical id, `url` = current physical endpoint |
| `GET` | `/control/workers/{id}` | `200` worker object · `404` |
| `GET` | `/control/workers/{id}/load` | `200` `{id,prefill_tokens,decode_blocks,inflight}` · `404` |

### 4.3 Router-side profiling (singleton resource + records sub-collection)
`profiling` is a resource whose `enabled` state you **set** (idempotent), and
whose captured `records` are a sub-collection you read or clear.

| Verb | Resource | Body | → |
|---|---|---|---|
| `GET` | `/control/profiling` | — | `200` `{enabled, since, record_count}` |
| `PUT` | `/control/profiling` | `{"enabled": true｜false}` | `200` `{enabled, since, record_count}` (idempotent start/stop) · `400` |
| `GET` | `/control/profiling/records` | `?last_n=N` (query) | `200` `{records:[…]}` |
| `DELETE` | `/control/profiling/records` | — | `204` (cleared) |
| `GET` | `/control/profiling/stats` | — | `200` aggregated snapshot |

*(Replaces the RPC `POST /profile {action}`: start/stop → `PUT …/profiling`,
clear → `DELETE …/profiling/records`, dump → `GET …/profiling/records`.)*

### 4.4 Per-worker GPU/torch profiling (sub-resource; proxied to engine)
Model each worker's profiler as a `profiling` sub-resource; `PUT enabled` proxies
to the engine's `/start_profile` / `/stop_profile`.

| Verb | Resource | Body | → |
|---|---|---|---|
| `PUT` | `/control/workers/{id}/profiling` | `{"enabled":true, ...profile_opts}` / `{"enabled":false}` | `200` `{id,enabled}` · `404` · `502` |
| `PUT` | `/control/workers/profiling` | `{"enabled":bool, ...}` | `200` `{results:[{id,ok}]}` · `502` (fleet) |

### 4.5 Runtime config
| Verb | Resource | Body | → |
|---|---|---|---|
| `GET` | `/control/config/log-level` | — | `200` `{level}` |
| `PUT` | `/control/config/log-level` | `{"level":"info,sgl_router::policies=trace"}` | `200` `{level}` · `400` (live `EnvFilter` reload, no restart) |
| `POST` | `/control/config:reload` | — | `202` `{reloaded:true,...}` — **P3**, non-idempotent action (re-read workers/policy tuning) |

### 4.6 Cache
Clearing a cache is a proxied **operation** on the worker's cache subsystem (it
maps to the engine's `POST /flush_cache`), so it's a `POST …/cache/clean` action
rather than a `DELETE` on a router resource. `GET` reads status.

| Verb | Resource | → |
|---|---|---|
| `GET` | `/control/cache` | `200` per-worker cache status (proxy, best-effort) |
| `POST` | `/control/cache/clean` | `200` `{results:[{id,ok}]}` · `502` (fleet; supersedes `POST /flush_cache`, kept as a back-compat alias) |
| `POST` | `/control/workers/{id}/cache/clean` | `200` `{id,ok}` · `404` · `502` (one worker) |

## 5. Architecture

### 5.1 Class diagram

```mermaid
classDiagram
    class ControlEndpoint {
        <<trait>>
        +id() &'static str
        +routes() Router~Arc~AppContext~~
    }

    class ControlPlane {
        -providers: Vec~Box~dyn ControlEndpoint~~
        +register(p: Box~dyn ControlEndpoint~)
        +into_router(auth: AuthLayer) Router
    }

    class AuthLayer {
        <<tower middleware>>
        -token: Option~String~
        +check(Authorization) Result
    }

    class DiagnosticsEndpoints {
        +/control/status
        +/control/workers
        +/control/workers/:id/load
    }
    class LogLevelEndpoints {
        -reload_handle: tracing_subscriber reload::Handle
        +GET/PUT /control/config/log-level
    }
    class ProfileEndpoints {
        -profiler: Arc~ProfileController~
        +PUT /control/profiling
        +GET /control/profiling/stats|records
        +DELETE /control/profiling/records
    }
    class WorkerProfileProxy {
        -proxy: Arc~Proxy~
        -registry: Arc~WorkerRegistry~
        +PUT /control/workers/:id/profiling
        +PUT /control/workers/profiling
    }
    class CacheEndpoints {
        +GET /control/cache
        +POST /control/cache/clean
        +POST /control/workers/:id/cache/clean
    }

    class ProfileController {
        -enabled: AtomicBool
        -ring: Mutex~RingBuffer~Record~~
        +set(enabled: bool)
        +record(r)
        +stats() Json
        +records(last_n) Json
    }

    class AppContext {
        +registry: Arc~WorkerRegistry~
        +proxy: Arc~Proxy~
        +metrics: Arc~MetricsRegistry~
        +control: Option~Arc~ControlPlane~~
    }

    ControlEndpoint <|.. DiagnosticsEndpoints
    ControlEndpoint <|.. LogLevelEndpoints
    ControlEndpoint <|.. ProfileEndpoints
    ControlEndpoint <|.. WorkerProfileProxy
    ControlEndpoint <|.. CacheEndpoints
    ControlPlane o--> "0..*" ControlEndpoint : registered
    ProfileEndpoints --> ProfileController : reads/toggles
    WorkerProfileProxy --> AppContext : proxy + registry
    AppContext --> ControlPlane : builds when enabled
    ControlPlane --> AuthLayer : wraps all routes
```

Notes:
- `ControlEndpoint::routes()` returns an axum sub-`Router` so each provider owns
  its paths; `ControlPlane::into_router` nests them under `/control` and wraps the
  whole thing in `AuthLayer`. Same "module self-registers when active" model as
  `MetricCollector`.
- Only providers whose feature is active are registered (e.g. `ProfileEndpoints`
  only when the profiler feature is on), so the surface is feature-driven.

### 5.2 Sequence — startup (control plane assembled only when enabled)

```mermaid
sequenceDiagram
    participant main
    participant Cli
    participant CP as ControlPlane
    participant App as build_router

    main->>Cli: parse() → Config
    alt --enable-control-api (or SGL_ROUTER_CONTROL)
        main->>CP: new()
        main->>CP: register(Diagnostics), register(LogLevel), register(Cache)
        opt profiler feature on
            main->>CP: register(ProfileEndpoints(ProfileController))
        end
        opt PD / workers present
            main->>CP: register(WorkerProfileProxy)
        end
        main->>CP: into_router(AuthLayer{token: SGL_ROUTER_CONTROL_TOKEN})
        CP-->>main: Router (/control/*)
        main->>App: merge control router (same port, or bind --control-addr)
    else disabled (default)
        Note over main,App: no /control/* routes exist at all
    end
```

### 5.3 Sequence — toggle router profiling + dump

```mermaid
sequenceDiagram
    participant Op as Operator
    participant Auth as AuthLayer
    participant PE as ProfileEndpoints
    participant PC as ProfileController
    participant Chat as chat_completions

    Op->>Auth: PUT /control/profiling {enabled:true} (Bearer)
    Auth->>PE: authorized
    PE->>PC: set(true)  (enabled=true)
    Note over Chat,PC: while enabled, the request path records into the ring
    Chat->>PC: record(per-request/session sample)
    Op->>Auth: GET /control/profiling/records?last_n=100 (Bearer)
    Auth->>PE: authorized
    PE->>PC: records(100)
    PC-->>Op: 200 {records:[ …recent samples… ]}
    Op->>Auth: PUT /control/profiling {enabled:false}
    Auth->>PE: set(false) (enabled=false)
```

### 5.4 Sequence — per-worker GPU/torch profile (proxy to engine)

```mermaid
sequenceDiagram
    participant Op as Operator
    participant WP as WorkerProfileProxy
    participant Reg as WorkerRegistry
    participant Eng as SGLang engine

    Op->>WP: PUT /control/workers/p0/profiling {enabled:true, ...opts}
    WP->>Reg: get("p0") → Worker{url}
    WP->>Eng: POST {url}/start_profile {ProfileReq}
    Eng-->>WP: 200
    WP-->>Op: 200 {id:"p0", enabled:true}
    Note over Op,Eng: later
    Op->>WP: PUT /control/workers/p0/profiling {enabled:false}
    WP->>Eng: POST {url}/stop_profile
    Eng-->>WP: 200 (trace written on the engine host)
```

### 5.5 Sequence — runtime log level

```mermaid
sequenceDiagram
    participant Op as Operator
    participant LL as LogLevelEndpoints
    participant H as tracing reload::Handle

    Op->>LL: PUT /control/config/log-level {level:"info,sgl_router::policies=trace"}
    LL->>H: reload(EnvFilter::new(level))
    H-->>LL: ok
    LL-->>Op: {applied:"info,sgl_router::policies=trace"}
    Note over H: subsequent logs use the new filter, no restart
```

Requires initializing the subscriber in `main.rs` with a
`tracing_subscriber::reload::Layer` and storing the `Handle` (on `AppContext` or
captured by `LogLevelEndpoints`). This is the one core change to `main.rs`.

### 5.6 mori-trace compatibility

`mori-trace` scrapes each worker's `/metrics` on an interval and discovers
targets from the scheduler's `/status`. sgl-router already serves `/metrics`
(now including per-session signals via the pluggable collectors), so:

- `GET /control/status` returns a targets list in a shape `mori-trace` accepts
  (`{workers:[{url}...]}`), letting `mori-trace --mori http://<sgl-router>` work
  unchanged;
- a dedicated `sgl-trace` binary is **not** required for v1 — reuse `mori-trace`.

## 6. Security

Control endpoints change runtime behavior and can start profilers on GPUs, so:

- **Disabled by default.** No `/control/*` route exists unless
  `--enable-control-api` / `SGL_ROUTER_CONTROL` is set.
- **Bearer auth.** When enabled, `SGL_ROUTER_CONTROL_TOKEN` is required; requests
  without a matching `Authorization: Bearer` get 401. Startup refuses to enable
  the control plane if the token is unset (fail-closed).
- **Optional separate bind.** `--control-addr` serves `/control/*` on its own
  listener (e.g. localhost / mesh-only), keeping it off the public data port.
- **Least surface.** Only the providers for active features register; read-only
  diagnostics are GET, mutations are POST/PUT.

## 7. Phasing

- **P1 (core, low-risk):** `ControlPlane` + `AuthLayer` + enable flag/token;
  `GET /control/status`, `/control/workers`; `GET/PUT /control/config/log-level`
  (reload handle in `main.rs`); `POST /control/cache/clean` (reuses existing
  fan-out); logical-id resolution (§3.1) + `GET /control/workers`. No engine changes.
- **P2 (profiling):** `ProfileController` + `PUT /control/profiling` +
  `GET /control/profiling/{stats,records}` + `DELETE /control/profiling/records`;
  per-worker proxy `PUT /control/workers/:id/profiling` → engine
  `/start_profile` `/stop_profile`.
- **P3 (config reload):** best-effort `POST /control/config:reload` (re-read
  worker URLs / policy tuning). Bounded scope; may stay out if risky.

## 8. Files (when implemented — not in this branch)

```
server/control/mod.rs           ControlEndpoint trait + ControlPlane + AuthLayer
server/control/diagnostics.rs   /control/status, /workers, /workers/:id/load
server/control/log_level.rs     /control/config/log-level (needs reload handle)
server/control/profile.rs       ProfileController + /control/profiling*
server/control/worker_proxy.rs  /control/workers/:id/profiling, /control/workers/:id/cache/clean
workers/registry.rs             logical-id ↔ WorkerId map + resolver (§3.1)
server/app.rs                   nest ControlPlane router when enabled
main.rs                         reload-layer subscriber; build ControlPlane
config/cli.rs, config/types.rs  --enable-control-api, --control-addr, token(env)
```

## 9. Non-goals / open questions

- No scheduling/placement/eviction logic — ops/observability only.
- Auth is a static bearer token (env). mTLS / RBAC is out of scope.
- `POST /control/config:reload` scope (workers only? policies too?) — TBD in P3.
- Whether to reuse the data port (nested router + auth) or require
  `--control-addr` in production — default: same port + auth; recommend separate
  bind for exposed deployments.
