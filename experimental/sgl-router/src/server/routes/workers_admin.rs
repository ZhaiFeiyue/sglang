// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-worker admin endpoints (v1): profiling start/end and cache clean.
//!
//! Each endpoint addresses a single worker by its registry `{id}` and **proxies
//! the action to that worker's SGLang engine** — the router holds no profiler
//! state and never handles trace files (torch traces are written on the engine
//! host; see the engine's `output_dir` / `SGLANG_TORCH_PROFILER_DIR`). The
//! router only triggers the action and reports the outcome.
//!
//! | Method | Path | Engine call |
//! |---|---|---|
//! | POST | `/workers/{id}/profiling/start` | `POST {url}/start_profile` (body forwarded) |
//! | POST | `/workers/{id}/profiling/end`   | `POST {url}/stop_profile` |
//! | POST | `/workers/{id}/cache/clean`     | `POST {url}/flush_cache` |
//!
//! Like `/flush_cache`, these bypass the circuit breaker (out-of-band admin
//! calls shouldn't skew the routing breaker state).

use crate::discovery::WorkerId;
use crate::server::app_context::AppContext;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::sync::Arc;

/// `POST /workers/{id}/profiling/start` — start torch profiling on the worker.
/// The request body (an optional SGLang `ProfileReq`, e.g. `{output_dir,
/// activities, num_steps, profile_id}`) is forwarded verbatim to the engine.
pub async fn profiling_start(
    State(ctx): State<Arc<AppContext>>,
    Path(id): Path<String>,
    body: Bytes,
) -> Response {
    proxy_worker_action(&ctx, &id, "start_profile", Some(body)).await
}

/// `POST /workers/{id}/profiling/end` — stop torch profiling; the engine writes
/// the trace to its configured `output_dir` on the engine host.
pub async fn profiling_end(State(ctx): State<Arc<AppContext>>, Path(id): Path<String>) -> Response {
    proxy_worker_action(&ctx, &id, "stop_profile", None).await
}

/// `POST /workers/{id}/cache/clean` — flush the worker's KV/prefix cache.
pub async fn cache_clean(State(ctx): State<Arc<AppContext>>, Path(id): Path<String>) -> Response {
    proxy_worker_action(&ctx, &id, "flush_cache", None).await
}

/// Resolve `{id}` to a worker and `POST {url}/{engine_path}`, forwarding
/// `body` when present. `404` for an unknown id, `200` on a 2xx upstream,
/// `502` on a non-2xx or transport error. The breaker is bypassed.
async fn proxy_worker_action(
    ctx: &AppContext,
    id: &str,
    engine_path: &str,
    body: Option<Bytes>,
) -> Response {
    let worker = match ctx.registry.get(&WorkerId(id.to_string())) {
        Some(w) => w,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "worker": id, "ok": false, "error": "unknown worker id" })),
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
            tracing::info!(worker = id, url = %worker.url, action = engine_path, "worker admin action ok");
            (
                StatusCode::OK,
                Json(json!({ "worker": id, "url": worker.url, "ok": true, "upstream_status": upstream })),
            )
                .into_response()
        }
        Ok(resp) => {
            let upstream = resp.status().as_u16();
            tracing::warn!(worker = id, url = %worker.url, action = engine_path, upstream, "worker admin action failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "worker": id, "url": worker.url, "ok": false, "upstream_status": upstream })),
            )
                .into_response()
        }
        Err(e) => {
            let err = format!("{:#}", anyhow::Error::new(e));
            tracing::warn!(worker = id, url = %worker.url, action = engine_path, error = %err, "worker admin action errored");
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "worker": id, "url": worker.url, "ok": false, "error": err })),
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

    fn ctx_with_worker(id: &str, url: &str) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId(id.to_string()),
                url: url.to_string(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: None,
            })
            .expect("worker accepted");
        Arc::new(ctx)
    }

    async fn call(ctx: Arc<AppContext>, uri: &str) -> (StatusCode, Value) {
        let app = crate::server::app::build_router(ctx);
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
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
    async fn profiling_and_cache_ok() {
        let (url, _s) = spawn_fake_engine(StatusCode::OK).await;
        let ctx = ctx_with_worker("prefill-0", &url);
        for path in [
            "/workers/prefill-0/profiling/start",
            "/workers/prefill-0/profiling/end",
            "/workers/prefill-0/cache/clean",
        ] {
            let (status, body) = call(ctx.clone(), path).await;
            assert_eq!(status, StatusCode::OK, "{path}");
            assert_eq!(body["ok"], true, "{path}: {body}");
            assert_eq!(body["worker"], "prefill-0");
        }
    }

    #[tokio::test]
    async fn unknown_worker_is_404() {
        let ctx = Arc::new(AppContext::stub());
        let (status, body) = call(ctx, "/workers/nope/cache/clean").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["ok"], false);
    }

    #[tokio::test]
    async fn upstream_failure_is_502() {
        let (url, _s) = spawn_fake_engine(StatusCode::INTERNAL_SERVER_ERROR).await;
        let ctx = ctx_with_worker("d0", &url);
        let (status, body) = call(ctx, "/workers/d0/profiling/start").await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["ok"], false);
        assert_eq!(body["upstream_status"], 500);
    }
}
