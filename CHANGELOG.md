# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Aligned packaging metadata around `pyproject.toml` as the single source of truth.
- Updated Python support metadata to target Python 3.10 through 3.12.
- Updated CI workflows to run against monorepo packages (`autoperfpy`, `trackiq_core`, `minicluster`, `trackiq_compare`).
- Expanded default coverage commands/config to include all monorepo packages.
- Refreshed `requirements.txt` to remove stale transitive pins and align with project metadata.
- Updated `Makefile` commands and cleanup paths to use current TrackIQ monorepo naming.

### Fixed
- Removed stale documentation references to files that no longer exist.
- Replaced garbled text/encoding artifacts in maintained files.
- Corrected repository link in `trackiq_core` dashboard footer.

## [1.0.0] - 2026-02-21

### Added
- TrackIQ monorepo structure with shared `trackiq_core` library.
- Tool packages:
  - `autoperfpy` for inference and edge benchmarking
  - `minicluster` for distributed training validation
  - `trackiq_compare` for canonical result comparison
- Canonical `TrackiqResult` schema and cross-tool serialization/validation workflow.
- CLI and Streamlit dashboards for each tool.
