/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::process::Command;
use std::process::Stdio;
use std::time::Duration;

use nix::sys::signal::Signal;
use nix::sys::signal::{self};
use nix::unistd::Pid;
use tokio::time::sleep;
use tokio::time::timeout;

/// Integration test for ProcessAllocator child process cleanup behavior.
/// Tests that when a ProcessAllocator parent process is killed, its children
/// are properly cleaned up.
#[tokio::test]
async fn test_process_allocator_child_cleanup() {
    let test_binary_path = buck_resources::get("monarch/hyperactor_mesh/test_bin").unwrap();
    println!("Starting test process allocator at: {:?}", test_binary_path);

    // Start the test process allocator
    let mut child = Command::new(&test_binary_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start test process allocator");

    let parent_pid = child.id();
    println!("Parent process started with PID: {}", parent_pid);

    // Give the parent time to spawn its children
    // We'll monitor the output to see when children are running
    #[allow(clippy::disallowed_methods)]
    sleep(Duration::from_secs(10)).await;

    // Get the list of child processes before killing the parent
    let children_before = get_child_processes(parent_pid);
    println!("Children before kill: {:?}", children_before);

    // Ensure we have some children processes
    assert!(
        !children_before.is_empty(),
        "Expected child processes to be spawned, but none were found"
    );

    // Kill the parent process with SIGKILL
    println!("Killing parent process with PID: {}", parent_pid);
    child.kill().unwrap();

    // Wait for the parent to be killed (in a blocking task since std::process::Child::wait is blocking)
    let wait_future = tokio::task::spawn_blocking(move || child.wait());

    let result = timeout(Duration::from_secs(5), wait_future).await;
    match result {
        Ok(Ok(exit_status)) => match exit_status {
            Ok(status) => println!("Parent process exited with status: {:?}", status),
            Err(e) => println!("Error waiting for parent process: {}", e),
        },
        Ok(Err(_)) => println!("Error in spawn_blocking task"),
        Err(_) => {
            println!("Parent process did not exit within timeout");
        }
    }

    // Give some time for children to be cleaned up
    #[allow(clippy::disallowed_methods)]
    sleep(Duration::from_secs(2)).await;

    // Check if children are still running
    let children_after = get_child_processes(parent_pid);
    println!("Children after kill: {:?}", children_after);

    // Check detailed process information for each child
    for child_pid in &children_before {
        let process_info = get_detailed_process_info(*child_pid);
        println!("Child {} detailed info: {:?}", child_pid, process_info);
    }

    // Wait much longer to see if children eventually exit due to channel hangup
    println!("Waiting longer to see if children eventually exit due to channel hangup...");
    for i in 1..=12 {
        // Wait up to 60 more seconds (5 second intervals)
        #[allow(clippy::disallowed_methods)]
        sleep(Duration::from_secs(5)).await;
        let remaining = children_before
            .iter()
            .filter(|&&pid| is_process_running(pid))
            .count();
        println!(
            "After {} seconds: {} children still running",
            i * 5,
            remaining
        );

        if remaining == 0 {
            println!("SUCCESS: All children eventually exited due to channel hangup!");
            break;
        }
    }

    // Final check
    let children_final = get_child_processes(parent_pid);
    println!("Children final check: {:?}", children_final);

    // Verify that all children have been cleaned up
    for child_pid in &children_before {
        let is_running = is_process_running(*child_pid);
        if is_running {
            println!(
                "WARNING: Child process {} is still running after parent was killed",
                child_pid
            );
            let process_info = get_detailed_process_info(*child_pid);
            println!(
                "Child {} detailed info after wait: {:?}",
                child_pid, process_info
            );
        }
    }

    // The test passes if children are cleaned up
    // If children are still running, we'll print warnings but not fail immediately
    // since cleanup might take a bit more time
    let remaining_children: Vec<_> = children_before
        .iter()
        .filter(|&&pid| is_process_running(pid))
        .collect();

    if !remaining_children.is_empty() {
        println!("=== TEST RESULT ===");
        println!("ProcessAllocator child cleanup test FAILED:");
        println!("Expected all child processes to be cleaned up when parent is killed,");
        println!(
            "but {} children are still running after 65+ seconds: {:?}",
            remaining_children.len(),
            remaining_children
        );

        // Clean up the remaining children manually to be good citizens
        for &&pid in &remaining_children {
            println!("Manually killing child process: {}", pid);
            let _ = signal::kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
        }

        panic!(
            "ProcessAllocator child cleanup test failed - children were not cleaned up even after extended wait"
        );
    }
}

/// Get the list of child processes for a given parent PID
fn get_child_processes(parent_pid: u32) -> Vec<u32> {
    let output = Command::new("pgrep")
        .args(["-P", &parent_pid.to_string()])
        .output();

    match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .collect(),
        _ => Vec::new(),
    }
}

/// Check if a process with the given PID is still running
fn is_process_running(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Get detailed process information for debugging
fn get_detailed_process_info(pid: u32) -> Option<String> {
    Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid,ppid,command"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                Some(String::from_utf8_lossy(&output.stdout).to_string())
            } else {
                None
            }
        })
}
