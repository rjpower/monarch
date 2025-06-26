/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// Test binary for ProcessAllocator child process cleanup behavior.
/// This binary creates a ProcessAllocator and spawns several child processes,
/// then keeps running until killed. It's designed to test whether child
/// processes are properly cleaned up when the parent process is killed.
use std::time::Duration;

use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::AllocConstraints;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::ProcState;
use hyperactor_mesh::alloc::ProcessAllocator;
use ndslice::shape;
use tokio::process::Command;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("ProcessAllocator test binary starting...");

    // Create a ProcessAllocator using the bootstrap binary
    let bootstrap_path = buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap();

    let cmd = Command::new(&bootstrap_path);
    let mut allocator = ProcessAllocator::new(cmd);

    // Create an allocation with 4 child processes
    let mut alloc = allocator
        .allocate(AllocSpec {
            shape: shape! { replica = 4 },
            constraints: AllocConstraints::default(),
        })
        .await?;

    println!("Allocation created, waiting for children to start...");

    // Wait for all children to be running
    let mut running_count = 0;
    while running_count < 4 {
        match alloc.next().await {
            Some(ProcState::Created { proc_id, coords }) => {
                println!("Child process created: {:?} at {:?}", proc_id, coords);
            }
            Some(ProcState::Running { proc_id, addr, .. }) => {
                println!("Child process running: {:?} at {:?}", proc_id, addr);
                running_count += 1;
            }
            Some(ProcState::Stopped { proc_id, reason }) => {
                println!("Child process stopped: {:?}, reason: {:?}", proc_id, reason);
            }
            Some(ProcState::Failed {
                world_id,
                description,
            }) => {
                println!(
                    "Allocation failed: {:?}, description: {}",
                    world_id, description
                );
                return Err(format!("Allocation failed: {}", description).into());
            }
            None => {
                println!("No more allocation events");
                break;
            }
        }
    }

    println!(
        "All {} children are running. Parent PID: {}",
        running_count,
        std::process::id()
    );

    // Keep the process running indefinitely
    // In the test, we'll kill this process and check if children are cleaned up
    loop {
        sleep(Duration::from_secs(1)).await;
        println!("Parent process still alive, children should be running...");
    }
}
