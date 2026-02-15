# Nullscope Roadmap

## Version 0.1.0 (Completed)

- [x] Zero-cost no-op pattern
- [x] Enabled context with scope hierarchy
- [x] Context vars for async safety
- [x] Pluggable reporter protocol
- [x] SimpleReporter for development
- [x] OpenTelemetry adapter
- [x] `py.typed` marker for PEP 561
- [x] Basic test coverage

## Version 0.2.0 (Current) - Ergonomics

- [x] Decorator support (`@telemetry.timed("operation")`)
- [x] Reporter lifecycle methods (`flush()`, `shutdown()`)
- [x] Async documentation and explicit testing
- [x] Improved error messages
- [x] Performance benchmarks (baseline + optimization validation)

## Version 0.3.0 - Observability

- [x] Exception tracking in scopes (record exception info)
- [x] Logging-based reporter adapter
- ~~Scope tags/labels support~~ — deferred (`**metadata` covers the use case)

## Version 0.4.0 - Polish

- [ ] Real-world usage feedback incorporated
- [ ] API refinements based on feedback

## Version 1.0.0 - Stable

- [ ] API freeze
- [ ] Comprehensive documentation
- [ ] Stability commitment
