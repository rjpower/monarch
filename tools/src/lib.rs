/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod args;
pub mod commands;

use clap::Parser;
use commands::Command;

#[derive(Parser, Debug)]
#[command(about = "Command line tool for monarch")]
pub struct Cli {
    #[command(subcommand)]
    pub subcmd: Command,
}
