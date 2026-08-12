// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-session request arrival-rate tracking + its pluggable `/metrics`
//! collector.
//!
//! Measurement-only: it records when requests arrive per session and exposes
//! each session's smoothed arrival rate as a Prometheus gauge. It classifies
//! nothing (no busy/idle) — a downstream consumer does that.
//!
//! This is an example of the self-registration pattern (see
//! [`crate::server::metrics_collector`]): the module is a self-contained data
//! source (`SessionArrivalTracker`) plus a [`MetricCollector`]
//! (`SessionArrivalRateCollector`). When the feature is enabled (env
//! `SGL_ROUTER_SESSION_ARRIVAL`), `AppContext` builds the tracker and registers
//! the collector, so the `sgl_router_session_arrival_rate` series appears on
//! `/metrics`; otherwise nothing is constructed and there is zero overhead.
//!
//! # Cardinality
//! The gauge carries a `session_id` label — that is inherently high-cardinality.
//! Idle sessions are evicted after a TTL (via [`SessionArrivalTracker::sweep`],
//! driven by the shared janitor) to bound the live label set. Intended for a
//! bounded session population (benchmarks / controlled deployments).

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::server::metrics_collector::MetricCollector;

/// Env that enables the feature (presence, non-empty). When unset the tracker is
/// not built and the collector is not registered.
const ENABLE_ENV: &str = "SGL_ROUTER_SESSION_ARRIVAL";
/// Sliding window of recent arrivals per session used to smooth the rate.
const WINDOW_ENV: &str = "SGL_ROUTER_SESSION_ARRIVAL_WINDOW";
/// Idle eviction TTL (seconds) — bounds the per-session label cardinality.
const TTL_ENV: &str = "SGL_ROUTER_SESSION_ARRIVAL_TTL_SECS";

const DEFAULT_WINDOW: usize = 8;
const DEFAULT_TTL_SECS: u64 = 900;

/// Live arrival state for one session: the last `window` arrival instants and
/// the last-seen instant (for TTL eviction).
#[derive(Debug)]
struct ArrivalState {
    recent: VecDeque<Instant>,
    last_seen: Instant,
}

/// Concurrent per-session arrival tracker.
#[derive(Debug)]
pub struct SessionArrivalTracker {
    sessions: DashMap<String, ArrivalState>,
    window: usize,
    ttl: Duration,
}

impl SessionArrivalTracker {
    /// Build from the environment, or `None` when the feature is disabled
    /// (`SGL_ROUTER_SESSION_ARRIVAL` unset/empty).
    pub fn from_env() -> Option<Arc<Self>> {
        let enabled = std::env::var(ENABLE_ENV)
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        if !enabled {
            return None;
        }
        let window = env_parse(WINDOW_ENV, DEFAULT_WINDOW).max(2);
        let ttl = Duration::from_secs(env_parse(TTL_ENV, DEFAULT_TTL_SECS));
        tracing::info!(
            window,
            ttl_secs = ttl.as_secs(),
            "session-arrival: per-session arrival-rate metric enabled"
        );
        Some(Arc::new(Self {
            sessions: DashMap::new(),
            window,
            ttl,
        }))
    }

    /// Record a request arrival for `sid` at `now`.
    pub fn on_recv(&self, sid: &str, now: Instant) {
        let mut e = self
            .sessions
            .entry(sid.to_string())
            .or_insert_with(|| ArrivalState {
                recent: VecDeque::with_capacity(self.window),
                last_seen: now,
            });
        e.recent.push_back(now);
        while e.recent.len() > self.window {
            e.recent.pop_front();
        }
        e.last_seen = now;
    }

    /// Evict sessions idle for longer than the TTL. Returns the count removed.
    /// Driven by `active_load::spawn_sweeper`.
    pub fn sweep(&self) -> usize {
        let now = Instant::now();
        let ttl = self.ttl;
        let before = self.sessions.len();
        self.sessions
            .retain(|_, e| now.saturating_duration_since(e.last_seen) <= ttl);
        before - self.sessions.len()
    }

    /// Snapshot `(session_id, arrival_rate_hz)` for every session with enough
    /// samples to compute a rate (≥ 2 arrivals in the window). `now` anchors the
    /// most recent interval so a session that has gone quiet decays toward 0.
    fn snapshot(&self, now: Instant) -> Vec<(String, f64)> {
        let mut out = Vec::with_capacity(self.sessions.len());
        for kv in self.sessions.iter() {
            if let Some(rate) = arrival_rate_hz(kv.value(), now) {
                out.push((kv.key().clone(), rate));
            }
        }
        out
    }
}

/// Smoothed arrival rate (turns/s) over the window: `(n - 1)` intervals across
/// `last - first` seconds. Extends the span to `now` so a session that stopped
/// arriving reports a decaying (lower) rate rather than a stale high one.
/// Returns `None` with fewer than 2 samples (no rate yet).
fn arrival_rate_hz(state: &ArrivalState, now: Instant) -> Option<f64> {
    let n = state.recent.len();
    if n < 2 {
        return None;
    }
    let first = *state.recent.front().unwrap();
    let last = *state.recent.back().unwrap();
    // Span from the oldest sample to max(most-recent sample, now): the trailing
    // gap since the last arrival counts against the rate (decay when quiet).
    let span = now.saturating_duration_since(first).as_secs_f64();
    let span = span.max(last.saturating_duration_since(first).as_secs_f64());
    if span <= 0.0 {
        return None;
    }
    Some((n as f64 - 1.0) / span)
}

/// Pluggable collector exposing `sgl_router_session_arrival_rate{session_id}`.
/// Holds the tracker and reads live values at render time.
#[derive(Debug)]
pub struct SessionArrivalRateCollector {
    tracker: Arc<SessionArrivalTracker>,
}

impl SessionArrivalRateCollector {
    pub const ID: &'static str = "session_arrival_rate";

    pub fn new(tracker: Arc<SessionArrivalTracker>) -> Self {
        Self { tracker }
    }
}

impl MetricCollector for SessionArrivalRateCollector {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn render(&self, out: &mut String) {
        out.push_str(
            "# HELP sgl_router_session_arrival_rate Per-session request arrival rate (turns/second), smoothed over recent arrivals.\n",
        );
        out.push_str("# TYPE sgl_router_session_arrival_rate gauge\n");
        for (sid, rate) in self.tracker.snapshot(Instant::now()) {
            out.push_str(&format!(
                "sgl_router_session_arrival_rate{{session_id=\"{}\"}} {}\n",
                escape_label(&sid),
                rate,
            ));
        }
    }
}

/// Escape a Prometheus label value (`\\`, `"`, newline) per exposition format.
fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker(window: usize) -> Arc<SessionArrivalTracker> {
        Arc::new(SessionArrivalTracker {
            sessions: DashMap::new(),
            window,
            ttl: Duration::from_secs(900),
        })
    }

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn no_rate_with_single_arrival() {
        let t = tracker(8);
        let t0 = Instant::now();
        t.on_recv("s", t0);
        assert!(t.snapshot(t0).is_empty(), "one arrival → no rate yet");
    }

    #[test]
    fn rate_reflects_arrival_cadence() {
        let t = tracker(8);
        let t0 = Instant::now();
        // 5 arrivals, 1s apart → 4 intervals over 4s → 1 turn/s.
        for i in 0..5 {
            t.on_recv("busy", at(t0, i * 1000));
        }
        let snap = t.snapshot(at(t0, 4000));
        let (_, rate) = snap.iter().find(|(s, _)| s == "busy").unwrap();
        assert!((rate - 1.0).abs() < 0.05, "≈1 turn/s, got {rate}");

        // A slow session: 3 arrivals 20s apart → 2 intervals / 40s = 0.05/s.
        for i in 0..3 {
            t.on_recv("idle", at(t0, i * 20_000));
        }
        let snap = t.snapshot(at(t0, 40_000));
        let (_, rate) = snap.iter().find(|(s, _)| s == "idle").unwrap();
        assert!((rate - 0.05).abs() < 0.005, "≈0.05 turn/s, got {rate}");
    }

    #[test]
    fn window_bounds_samples() {
        let t = tracker(3);
        let t0 = Instant::now();
        for i in 0..10 {
            t.on_recv("s", at(t0, i * 1000));
        }
        // Only the last 3 arrivals are kept → 2 intervals over 2s = 1/s.
        let snap = t.snapshot(at(t0, 9000));
        let (_, rate) = snap.iter().find(|(s, _)| s == "s").unwrap();
        assert!((rate - 1.0).abs() < 0.05, "windowed rate ≈1/s, got {rate}");
    }

    #[test]
    fn sweep_evicts_idle() {
        let t = Arc::new(SessionArrivalTracker {
            sessions: DashMap::new(),
            window: 8,
            ttl: Duration::from_millis(0),
        });
        let now = Instant::now();
        t.on_recv("s", now);
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(t.sweep(), 1);
        assert!(t.sessions.is_empty());
    }

    #[test]
    fn render_emits_gauge_for_sessions_with_rate() {
        let t = tracker(8);
        let t0 = Instant::now();
        for i in 0..3 {
            t.on_recv("abc", at(t0, i * 1000));
        }
        let collector = SessionArrivalRateCollector::new(t);
        let mut out = String::new();
        collector.render(&mut out);
        assert!(out.contains("# TYPE sgl_router_session_arrival_rate gauge"));
        assert!(
            out.contains("sgl_router_session_arrival_rate{session_id=\"abc\"}"),
            "got:\n{out}"
        );
    }
}
