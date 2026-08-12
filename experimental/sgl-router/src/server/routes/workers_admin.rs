// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-worker admin endpoints (v1): worker listing, profiling start/end, and
//! cache clean.
//!
//! Workers are addressed by a small **integer id** (`0`, `1`, `2`, …) that is
//! **stable across restarts**: the id is the worker's rank when the registry is
//! sorted deterministically by URL, so the same set of worker URLs always yields
//! the same id → worker mapping regardless of discovery/restart order. Use
//! `GET /workers` to see the current id → url mapping.
//!
//! Each action **proxies to the addressed worker's SGLang engine** — the router
//! holds no profiler state and never handles trace files (torch traces are
//! written on the engine host; see the engine's `output_dir` /
//! `SGLANG_TORCH_PROFILER_DIR`). The router only triggers and reports.
//!
//! | Method | Path | Engine call |
//! |---|---|---|
//! | GET  | `/workers` | — (list `{id,url,mode}`) |
//! | POST | `/workers/{id}/profiling/start` | `POST {url}/start_profile` (body forwarded) |
//! | POST | `/workers/{id}/profiling/end`   | `POST {url}/stop_profile` |
//! | POST | `/workers/{id}/cache/clean`     | `POST {url}/flush_cache` |
//!
//! Like `/flush_cache`, the actions bypass the circuit breaker (out-of-band
//! admin calls shouldn't skew the routing breaker state).

use crate::discovery::WorkerMode;
use crate::server::app_context::AppContext;
use crate::workers::worker::Worker;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;

fn mode_str(mode: WorkerMode) -> &'static str {
    match mode {
        WorkerMode::Plain => "plain",
        WorkerMode::Prefill => "prefill",
        WorkerMode::Decode => "decode",
    }
}

/// Registry workers in the deterministic order that defines the integer ids:
/// sorted by URL. Index `i` in the returned vec is worker id `i`.
fn ordered_workers(ctx: &AppContext) -> Vec<Arc<Worker>> {
    let mut workers = ctx.registry.all();
    workers.sort_by(|a, b| a.url.cmp(&b.url));
    workers
}

/// One row of `GET /workers`.
#[derive(Serialize)]
pub struct WorkerEntry {
    id: usize,
    url: String,
    mode: &'static str,
}

/// `GET /workers` — list workers with their stable integer id, url, and mode,
/// so a caller knows which `{id}` to address.
pub async fn list_workers(State(ctx): State<Arc<AppContext>>) -> Json<Vec<WorkerEntry>> {
    let list = ordered_workers(&ctx)
        .iter()
        .enumerate()
        .map(|(id, w)| WorkerEntry {
            id,
            url: w.url.clone(),
            mode: mode_str(w.mode()),
        })
        .collect();
    Json(list)
}

/// `POST /workers/{id}/profiling/start` — start torch profiling on the worker.
/// The request body (an optional SGLang `ProfileReq`, e.g. `{output_dir,
/// activities, num_steps, profile_id}`) is forwarded verbatim to the engine.
pub async fn profiling_start(
    State(ctx): State<Arc<AppContext>>,
    Path(id): Path<usize>,
    body: Bytes,
) -> Response {
    proxy_worker_action(&ctx, id, "start_profile", Some(body)).await
}

/// `POST /workers/{id}/profiling/end` — stop torch profiling; the engine writes
/// the trace to its configured `output_dir` on the engine host.
pub async fn profiling_end(State(ctx): State<Arc<AppContext>>, Path(id): Path<usize>) -> Response {
    proxy_worker_action(&ctx, id, "stop_profile", None).await
}

/// `POST /workers/{id}/cache/clean` — flush the worker's KV/prefix cache.
pub async fn cache_clean(State(ctx): State<Arc<AppContext>>, Path(id): Path<usize>) -> Response {
    proxy_worker_action(&ctx, id, "flush_cache", None).await
}

/// Resolve the integer `id` to a worker and `POST {url}/{engine_path}`,
/// forwarding `body` when present. `404` for an out-of-range id, `200` on a 2xx
/// upstream, `502` on a non-2xx or transport error. The breaker is bypassed.
async fn proxy_worker_action(
    ctx: &AppContext,
    id: usize,
    engine_path: &str,
    body: Option<Bytes>,
) -> Response {
    let worker = match ordered_workers(ctx).into_iter().nth(id) {
        Some(w) => w,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "id": id, "ok": false, "error": "no worker with that id" })),
            )
                .into_response();
        }
    };

    let url = format!("{}/{}", worker.url.trim_end_matches('/'), engine_path);
    let mut req = ctx
        .proxy
        .client
        .post(&url)
        .timeout(ctx.proxy.request_timeout);
    if let Some(b) = body {
        if !b.is_empty() {
            req = req.header("content-type", "application/json").body(b);
        }
    }

    match req.send().await {
        Ok(resp) if resp.status().is_success() => {
            let upstream = resp.status().as_u16();
            tracing::info!(id, url = %worker.url, action = engine_path, "worker admin action ok");
            (
                StatusCode::OK,
                Json(
                    json!({ "id": id, "url": worker.url, "ok": true, "upstream_status": upstream }),
                ),
            )
                .into_response()
        }
        Ok(resp) => {
            let upstream = resp.status().as_u16();
            tracing::warn!(id, url = %worker.url, action = engine_path, upstream, "worker admin action failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "id": id, "url": worker.url, "ok": false, "upstream_status": upstream })),
            )
                .into_response()
        }
        Err(e) => {
            let err = format!("{:#}", anyhow::Error::new(e));
            tracing::warn!(id, url = %worker.url, action = engine_path, error = %err, "worker admin action errored");
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "id": id, "url": worker.url, "ok": false, "error": err })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::server::app_context::AppContext;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use std::sync::Arc;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tower::ServiceExt;

    /// Fake engine answering the three admin paths with `status`.
    async fn spawn_fake_engine(status: StatusCode) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new()
            .route("/start_profile", post(move || async move { status }))
            .route("/stop_profile", post(move || async move { status }))
            .route("/flush_cache", post(move || async move { status }));
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    fn add_worker(ctx: &AppContext, wid: &str, url: &str, mode: WorkerMode) {
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId(wid.to_string()),
                url: url.to_string(),
                mode,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: None,
            })
            .expect("worker accepted");
    }

    async fn call(ctx: Arc<AppContext>, method: &str, uri: &str) -> (StatusCode, Value) {
        let app = crate::server::app::build_router(ctx);
        let res = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn list_workers_returns_integer_ids_sorted_by_url() {
        let ctx = Arc::new(AppContext::stub());
        // Insert out of URL order; ids must still be url-sorted (0,1) and stable.
        add_worker(&ctx, "b", "http://10.0.0.2:31200", WorkerMode::Decode);
        add_worker(&ctx, "a", "http://10.0.0.1:31100", WorkerMode::Prefill);
        let (status, body) = call(ctx, "GET", "/workers").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body[0]["id"], 0);
        assert_eq!(body[0]["url"], "http://10.0.0.1:31100");
        assert_eq!(body[1]["id"], 1);
        assert_eq!(body[1]["url"], "http://10.0.0.2:31200");
    }

    #[tokio::test]
    async fn actions_resolve_worker_by_integer_id() {
        let (url0, _s0) = spawn_fake_engine(StatusCode::OK).await;
        let (url1, _s1) = spawn_fake_engine(StatusCode::OK).await;
        let ctx = Arc::new(AppContext::stub());
        // Ensure a deterministic url order: force url0 < url1 by sorting inputs.
        let (lo, hi) = if url0 < url1 {
            (url0.clone(), url1.clone())
        } else {
            (url1.clone(), url0.clone())
        };
        add_worker(&ctx, "w0", &lo, WorkerMode::Prefill);
        add_worker(&ctx, "w1", &hi, WorkerMode::Decode);

        for (idx, expect_url) in [(0usize, &lo), (1usize, &hi)] {
            let (status, body) =
                call(ctx.clone(), "POST", &format!("/workers/{idx}/cache/clean")).await;
            assert_eq!(status, StatusCode::OK, "id {idx}");
            assert_eq!(body["id"], idx);
            assert_eq!(body["url"], *expect_url);
            assert_eq!(body["ok"], true);
        }
    }

    #[tokio::test]
    async fn out_of_range_id_is_404() {
        let ctx = Arc::new(AppContext::stub());
        add_worker(&ctx, "w0", "http://10.0.0.1:31100", WorkerMode::Plain);
        let (status, body) = call(ctx, "POST", "/workers/5/profiling/start").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["ok"], false);
    }

    #[tokio::test]
    async fn non_integer_id_is_rejected() {
        let ctx = Arc::new(AppContext::stub());
        add_worker(&ctx, "w0", "http://10.0.0.1:31100", WorkerMode::Plain);
        // A non-numeric {id} fails the usize path extractor → 400 (plain-text
        // body, so we assert status only rather than parsing JSON).
        let app = crate::server::app::build_router(ctx);
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/workers/abc/cache/clean")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn upstream_failure_is_502() {
        let (url, _s) = spawn_fake_engine(StatusCode::INTERNAL_SERVER_ERROR).await;
        let ctx = Arc::new(AppContext::stub());
        add_worker(&ctx, "w0", &url, WorkerMode::Plain);
        let (status, body) = call(ctx, "POST", "/workers/0/profiling/start").await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["ok"], false);
        assert_eq!(body["upstream_status"], 500);
    }
}
