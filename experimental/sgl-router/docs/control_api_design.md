# Design — sgl-router control / profiling API

> **Status: design only (no implementation in this branch).** This document
> specifies a control-plane HTTP API for `sgl-router`, aligned with
> mori-scheduler's profiling/control surface, built on the same self-registration
> pattern as the pluggable `/metrics` collectors.

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
| Fleet cache flush | `/cache/flush` | `POST /flush_cache` | keep + `GET /control/cache/status`, per-worker |
| Router profiling toggle | `POST /profile/control` | — | `POST /control/profile` |
| Profiling stats / dump | `/profile/stats`, `/profile/dump` | — | `GET /control/profile/stats`, `/control/profile/dump` |
| Per-worker GPU/torch profile | `/workers/:id/profile/{start,stop}` | — | proxy → engine `/start_profile` `/stop_profile` |
| Runtime log level | `GET/PUT /router/config/log-level` | — | `GET/PUT /control/log-level` |
| Config reload | `POST /router/config/reload` | — | `POST /control/config/reload` (P3, best-effort) |
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

## 4. API surface

Base: `/control` (guarded). `{id}` is a `WorkerId` from the registry.

### 4.1 Diagnostics (read-only)
| Method | Path | Body / Query | Returns |
|---|---|---|---|
| GET | `/control/status` | — | cluster snapshot: router version, uptime, policy per model, worker count by mode, ready flag |
| GET | `/control/workers` | — | `[{id, url, mode, healthy, cb_state, inflight, model_ids}]` |
| GET | `/control/workers/{id}/load` | — | `{id, prefill_tokens, decode_blocks, inflight}` |

### 4.2 Profiling
| Method | Path | Body | Effect |
|---|---|---|---|
| POST | `/control/profile` | `{"action":"start"｜"stop"｜"clear"}` | toggle **router-side** capture (e.g. enable the session-signal collectors / an in-memory ring buffer) at runtime |
| GET | `/control/profile/stats` | — | aggregated router-side profiling snapshot (JSON) |
| GET | `/control/profile/dump` | `?last_n=N` | recent N per-request/session records (JSON) |
| POST | `/control/workers/{id}/profile/start` | `{...ProfileReq}` | proxy → engine `POST /start_profile` |
| POST | `/control/workers/{id}/profile/stop` | — | proxy → engine `POST /stop_profile` |
| POST | `/control/gpu_profile/{start,stop}` | `{worker?, ...}` | fan-out torch profiler to all (or one) worker |

### 4.3 Runtime config
| Method | Path | Body | Effect |
|---|---|---|---|
| GET | `/control/log-level` | — | current tracing filter directive |
| PUT | `/control/log-level` | `{"level":"debug"｜"info,sgl_router::policies=trace"}` | live `EnvFilter` reload (no restart) |
| POST | `/control/config/reload` | — | **P3**, best-effort re-read of worker list / policy tuning |

### 4.4 Cache
| Method | Path | Body | Effect |
|---|---|---|---|
| POST | `/control/cache/flush` | — | fleet flush (alias of today's `/flush_cache`) |
| GET | `/control/cache/status` | — | per-worker cache status (proxy → engine, best-effort) |
| POST | `/control/workers/{id}/cache/flush` | — | flush one worker |

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
        +GET/PUT /control/log-level
    }
    class ProfileEndpoints {
        -profiler: Arc~ProfileController~
        +POST /control/profile
        +GET /control/profile/stats|dump
    }
    class WorkerProfileProxy {
        -proxy: Arc~Proxy~
        -registry: Arc~WorkerRegistry~
        +/control/workers/:id/profile/*
        +/control/gpu_profile/*
    }
    class CacheEndpoints {
        +/control/cache/flush|status
        +/control/workers/:id/cache/flush
    }

    class ProfileController {
        -enabled: AtomicBool
        -ring: Mutex~RingBuffer~Record~~
        +set(action)
        +record(r)
        +stats() Json
        +dump(last_n) Json
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

    Op->>Auth: POST /control/profile {action:start} (Bearer)
    Auth->>PE: authorized
    PE->>PC: set(Start)  (enabled=true)
    Note over Chat,PC: while enabled, the request path records into the ring
    Chat->>PC: record(per-request/session sample)
    Op->>Auth: GET /control/profile/dump?last_n=100 (Bearer)
    Auth->>PE: authorized
    PE->>PC: dump(100)
    PC-->>Op: JSON [ …recent samples… ]
    Op->>Auth: POST /control/profile {action:stop}
    Auth->>PE: set(Stop) (enabled=false)
```

### 5.4 Sequence — per-worker GPU/torch profile (proxy to engine)

```mermaid
sequenceDiagram
    participant Op as Operator
    participant WP as WorkerProfileProxy
    participant Reg as WorkerRegistry
    participant Eng as SGLang engine

    Op->>WP: POST /control/workers/p0/profile/start {ProfileReq}
    WP->>Reg: get("p0") → Worker{url}
    WP->>Eng: POST {url}/start_profile {ProfileReq}
    Eng-->>WP: 200
    WP-->>Op: {worker:"p0", ok:true}
    Note over Op,Eng: later
    Op->>WP: POST /control/workers/p0/profile/stop
    WP->>Eng: POST {url}/stop_profile
    Eng-->>WP: 200 (trace written on the engine host)
```

### 5.5 Sequence — runtime log level

```mermaid
sequenceDiagram
    participant Op as Operator
    participant LL as LogLevelEndpoints
    participant H as tracing reload::Handle

    Op->>LL: PUT /control/log-level {level:"info,sgl_router::policies=trace"}
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
  `GET /control/status`, `/control/workers`; `GET/PUT /control/log-level`
  (reload handle in `main.rs`); `POST /control/cache/flush` (alias existing
  fan-out). No engine changes.
- **P2 (profiling):** `ProfileController` + `POST /control/profile`,
  `/control/profile/{stats,dump}`; per-worker proxy
  `/control/workers/:id/profile/{start,stop}` + `/control/gpu_profile/*` →
  engine `/start_profile` `/stop_profile`.
- **P3 (config reload):** best-effort `/control/config/reload` (re-read worker
  URLs / policy tuning). Bounded scope; may stay out if risky.

## 8. Files (when implemented — not in this branch)

```
server/control/mod.rs           ControlEndpoint trait + ControlPlane + AuthLayer
server/control/diagnostics.rs   /control/status, /workers, /workers/:id/load
server/control/log_level.rs     /control/log-level (needs reload handle)
server/control/profile.rs       ProfileController + /control/profile*
server/control/worker_proxy.rs  /control/workers/:id/profile|cache, /gpu_profile
server/app.rs                   nest ControlPlane router when enabled
main.rs                         reload-layer subscriber; build ControlPlane
config/cli.rs, config/types.rs  --enable-control-api, --control-addr, token(env)
```

## 9. Non-goals / open questions

- No scheduling/placement/eviction logic — ops/observability only.
- Auth is a static bearer token (env). mTLS / RBAC is out of scope.
- `/control/config/reload` scope (workers only? policies too?) — TBD in P3.
- Whether to reuse the data port (nested router + auth) or require
  `--control-addr` in production — default: same port + auth; recommend separate
  bind for exposed deployments.
