/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod bounce;
mod stop;

use clap::Subcommand;
use enum_dispatch::enum_dispatch;

/// Error that `Runnable` throws.
/// Currently type aliases to `anyhow::Error` but could
/// implement a custom error type in the future if needed.
type Error = anyhow::Error;

/// The run logic for each subcommand.
#[enum_dispatch]
pub trait Runnable {
    fn run(&self) -> Result<(), Error>;
}

/// Sub-commands of the CLI.
///
/// NOTE: create and teardown subcommands fallback to //monarch/tools:py_cli
///   since they use TorchX which only has a Python API.
#[derive(Subcommand, Debug)]
#[enum_dispatch(Runnable)]
pub enum Command {
    /// (re)starts the server's processes without tearing down the server instance
    #[command()]
    Bounce(bounce::Cmd),
    /// Stops the server's unix processes without tearing down the server instance
    #[command()]
    Stop(stop::Cmd),
}
