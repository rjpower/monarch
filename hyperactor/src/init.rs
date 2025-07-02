/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::OnceLock;

use crate::clock::ClockKind;
use crate::panic_handler;

/// A global runtime handle used for spawning tasks. Do not use for executing long running or
/// compute intensive tasks.
static RUNTIME: OnceLock<tokio::runtime::Handle> = OnceLock::new();

/// Get a handle to the global runtime. Panics if hyperactor::initialize has not been called.
pub(crate) fn get_runtime() -> tokio::runtime::Handle {
    RUNTIME
        .get()
        .expect("hyperactor::initialize must be called")
        .clone()
}

/// Initialize the Hyperactor runtime. Specifically:
/// - Set up panic handling, so that we get consistent panic stack traces in Actors.
/// - Initialize logging defaults.
/// - Store the provided tokio runtime handle for use by the hyperactor system.
pub fn initialize(handle: tokio::runtime::Handle) {
    RUNTIME
        .set(handle)
        .expect("hyperactor::initialize must only be called once");

    panic_handler::set_panic_hook();
    hyperactor_telemetry::initialize_logging(ClockKind::default());
    #[cfg(target_os = "linux")]
    linux::initialize();
}

/// Initialize the Hyperactor runtime using the current tokio runtime handle.
/// This should be used when already within an async context where a runtime is running.
/// NEVER creates a duplicate runtime - uses the existing one.
/// - Set up panic handling, so that we get consistent panic stack traces in Actors.
/// - Initialize logging defaults.
/// - Use the current tokio runtime handle for the hyperactor system.
pub fn initialize_with_current() {
    // Get the current runtime handle - this will panic if not in an async context
    let handle = tokio::runtime::Handle::current();
    initialize(handle);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::backtrace::Backtrace;

    use nix::sys::signal::SigHandler;

    pub(crate) fn initialize() {
        // Safety: Because I want to
        unsafe {
            extern "C" fn handle_fatal_signal(signo: libc::c_int) {
                let bt = Backtrace::force_capture();
                let signame = nix::sys::signal::Signal::try_from(signo).expect("unknown signal");
                tracing::error!("stacktrace"= %bt, "fatal signal {signo}:{signame} received");
                std::process::exit(1);
            }
            nix::sys::signal::signal(
                nix::sys::signal::SIGABRT,
                SigHandler::Handler(handle_fatal_signal),
            )
            .expect("unable to register signal handler");
            nix::sys::signal::signal(
                nix::sys::signal::SIGSEGV,
                SigHandler::Handler(handle_fatal_signal),
            )
            .expect("unable to register signal handler");
        }
    }
}
