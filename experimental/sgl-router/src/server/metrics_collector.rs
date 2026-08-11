// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Pluggable `/metrics` collectors.
//!
//! The core [`crate::server::metrics::MetricsRegistry`] holds a fixed set of
//! always-on metric families. On top of that, any **module** can contribute its
//! own metrics by implementing [`MetricCollector`] and **registering itself**
//! (`MetricsRegistry::register_collector`) at the point where a param enables
//! its feature or selects it. That is what "pluggable" means here:
//!
//!   * enabling is driven by the feature/module, not a central metrics list —
//!     a module that isn't active never registers, so its series never appear;
//!   * the registrant owns *what* it emits;
//!   * `/metrics` just renders every registered collector in Prometheus format
//!     after the core families — it knows nothing about individual metrics.
//!
//! Adding metrics to a module is therefore local to that module: implement the
//! trait and call `register_collector` where the module is wired up (e.g. inside
//! its `attach_metrics`, or at construction when its CLI flag / env is set).
//! [`BuildInfoCollector`] below is a always-on sample that exercises the path.

/// One pluggable metric group. `render` appends this group's complete Prometheus
/// exposition (its own `# HELP` / `# TYPE` lines plus series) to the scrape
/// buffer. Implementors pull live values from their own data source at render
/// time — they don't push into the core registry.
pub trait MetricCollector: Send + Sync + std::fmt::Debug {
    /// Stable identifier (for logs / dedup). Not a user-facing toggle — whether
    /// a collector is present is decided by whether its module registered it.
    fn id(&self) -> &'static str;

    /// Append this collector's Prometheus text to `out`.
    fn render(&self, out: &mut String);
}

/// Always-on sample collector: emits the router build version. Registered
/// unconditionally in `AppContext` — its purpose is to exercise the whole
/// pluggable path (module registers → `/metrics` renders) end to end and to
/// give scrapers a build-identity series.
#[derive(Debug)]
pub struct BuildInfoCollector;

impl BuildInfoCollector {
    pub const ID: &'static str = "build_info";
}

impl MetricCollector for BuildInfoCollector {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn render(&self, out: &mut String) {
        out.push_str(
            "# HELP sgl_router_build_info Router build information (value is always 1).\n",
        );
        out.push_str("# TYPE sgl_router_build_info gauge\n");
        out.push_str(&format!(
            "sgl_router_build_info{{version=\"{}\"}} 1\n",
            crate::VERSION
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::MetricsRegistry;
    use std::sync::Arc;

    #[test]
    fn build_info_renders_series() {
        let mut out = String::new();
        BuildInfoCollector.render(&mut out);
        assert!(out.contains("# TYPE sgl_router_build_info gauge"));
        assert!(out.contains("sgl_router_build_info{version="));
    }

    #[test]
    fn registered_collector_appears_in_registry_render() {
        let metrics = MetricsRegistry::new();
        // Nothing registered → the collector's series is absent.
        assert!(!metrics
            .render_with_workers(&[])
            .contains("sgl_router_build_info"));
        // A module registers it → it renders after the core families.
        metrics.register_collector(Arc::new(BuildInfoCollector));
        let body = metrics.render_with_workers(&[]);
        assert!(body.contains("sgl_router_build_info{version="));
        // Core families still present (collectors are additive).
        assert!(body.contains("# TYPE sgl_router_requests_total counter"));
    }
}
