/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Utilities for launching hyperactor processes.

use std::sync::LazyLock;
use std::sync::OnceLock;

use crate::clock::ClockKind;
use crate::panic_handler;

/// A global runtime used in binding async and sync code. Do not use for executing long running or
/// compute intensive tasks.
pub(crate) static RUNTIME: LazyLock<tokio::runtime::Runtime> =
    LazyLock::new(|| tokio::runtime::Runtime::new().expect("failed to create global runtime"));

/// Initialize the Hyperactor runtime. Specifically:
/// - Set up panic handling, so that we get consistent panic stack traces in Actors.
/// - Initialize logging defaults.
pub fn initialize() {
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| {
        panic_handler::set_panic_hook();
        hyperactor_telemetry::initialize_logging(ClockKind::default());
    });
}
