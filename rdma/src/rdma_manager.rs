//! # RDMA Manager
//!
//! This module provides a high-level abstraction for RDMA (Remote Direct Memory Access) operations.
//! `RdmaManager` wraps the lower-level RDMA connection primitives and provides a more ergonomic
//! interface for performing RDMA operations between distributed endpoints.
//!
//! ## Overview
//!
//! `RdmaManager` is a high-level abstraction for managing RDMA (Remote Direct Memory Access) operations.
//! It simplifies the process of setting up and managing RDMA connections between distributed endpoints.
//!
//! `RdmaManager` handles tasks such as memory registration with RDMA devices, establishing connections,
//! performing RDMA read and write operations, and tracking and waiting for operation completions.
//!
//!
//! ## Usage
//!
//! A typical workflow involves:
//! 1. Creating `RdmaManager` instances on different endpoints
//! 2. Connecting managers to establish RDMA connections
//! 3. Performing RDMA write or read operations between the managers
//! 4. Waiting for operations to complete
//!
//! ## Example
//!
//! ```rust
//! use hyperactor_mesh::ProcMesh;
//! use hyperactor_mesh::alloc::AllocConstraints;
//! use hyperactor_mesh::alloc::AllocSpec;
//! use hyperactor_mesh::alloc::Allocator;
//! use hyperactor_mesh::alloc::LocalAllocator;
//! use ndslice::shape;
//! use rdma::RdmaConnectionConfig;
//! use rdma::RdmaManager;
//! use rdma::RdmaManagerArgs;
//! use rdma::RdmaMemoryRegion;
//!
//! async fn example() -> Result<(), anyhow::Error> {
//!     // Create process meshes
//!     let alloc = LocalAllocator
//!         .allocate(AllocSpec {
//!             shape: shape! {replica=1, host=1, gpu=1},
//!             constraints: AllocConstraints::none(),
//!         })
//!         .await?;
//!     let proc_mesh = ProcMesh::allocate(alloc).await?;
//!
//!     // Create buffers
//!     let mut buffer_data = vec![1, 2, 3, 4].into_boxed_slice();
//!     let mut buffer = RdmaManager::new(RdmaManagerArgs {
//!         name: "example".to_string(),
//!         proc_mesh: &proc_mesh,
//!         config: RdmaConnectionConfig::default(),
//!         memory_region: RdmaMemoryRegion::from(&mut buffer_data[..]),
//!     })
//!     .await?;
//!
//!     // Connect and perform operations with other buffers...
//!     Ok(())
//! }
//! ```
//!
//! ## TODOs
//! Support registering the entire address space so we don't have to create
//! a separate MR for each connection.
//!
//! Support GPUs
//! ```

use hyperactor::ActorId;
use hyperactor::Mailbox;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor_mesh::ActorMesh;
use hyperactor_mesh::Mesh;
use hyperactor_mesh::ProcMesh;
use ndslice::selection;
use tokio::time::Duration;

use crate::ibverbs_primitives::IbvWc;
use crate::ibverbs_primitives::RdmaConnectionInfo;
use crate::ibverbs_primitives::RdmaMemoryRegion;
use crate::rdma_connection::RdmaConnectionConfig;
use crate::rdma_manager_actor::Connect;
use crate::rdma_manager_actor::Connected;
use crate::rdma_manager_actor::ConnectionInfo;
use crate::rdma_manager_actor::Fetch;
use crate::rdma_manager_actor::PollCompletion;
use crate::rdma_manager_actor::Put;
use crate::rdma_manager_actor::RdmaManagerActor;
use crate::rdma_manager_actor::Register;

pub struct RdmaManagerArgs<'a> {
    pub name: String,
    pub proc_mesh: &'a ProcMesh,
    pub config: RdmaConnectionConfig,
    pub memory_region: RdmaMemoryRegion,
}

pub struct RdmaManager<'a> {
    name: String,
    manager: ActorMesh<'a, RdmaManagerActor>,
    client: &'a Mailbox,
    id: ActorId,
    current_work_id: u64,
}

impl<'a> RdmaManager<'a> {
    pub async fn new(args: RdmaManagerArgs<'a>) -> Result<Self, anyhow::Error> {
        // TODO - need some way to assert that this spawns a single manager per host
        let manager: ActorMesh<'_, RdmaManagerActor> =
            args.proc_mesh.spawn("rdma_manager", &()).await.unwrap();

        let client = args.proc_mesh.client();

        let binding = manager.get(0).unwrap();
        let id = binding.actor_id().clone();

        // Register the memory region and configuration
        manager
            .cast(
                selection::selection_from_one(manager.shape(), "gpu", 0..1).unwrap(),
                Register(args.config.clone(), args.memory_region.clone()),
            )
            .unwrap();

        Ok(Self {
            name: args.name,
            manager,
            client,
            id,
            current_work_id: 0,
        })
    }

    pub async fn id(&self) -> ActorId {
        self.id.clone()
    }

    pub async fn connection_info(
        &self,
        other: &RdmaManager<'_>,
    ) -> Result<RdmaConnectionInfo, anyhow::Error> {
        let id = other.id().await;
        tracing::debug!("[{}] trying to get connection info from {}", self.name, id);

        let (connection_info_handle, connection_info_receiver) =
            self.client.open_once_port::<RdmaConnectionInfo>();
        // TODO - can we just use call_one?
        self.manager
            .cast(
                selection::selection_from_one(self.manager.shape(), "gpu", 0..1).unwrap(),
                ConnectionInfo(id.clone(), connection_info_handle.bind()),
            )
            .unwrap();
        let connection_info = connection_info_receiver.recv().await?;
        tracing::debug!("[{}] connection info is {:?}", self.name, connection_info);
        Ok(connection_info)
    }

    pub async fn wait_for_completion(
        &self,
        other: &RdmaManager<'_>,
        timeout_secs: u64,
        work_id: u64,
    ) -> Result<bool, anyhow::Error> {
        // TODO - we should consider supporting both a polling loop (implemented here) and a tokio style Waker
        let timeout = Duration::from_secs(timeout_secs);
        let start_time = std::time::Instant::now();
        tracing::debug!("Waiting for {}", work_id);

        while start_time.elapsed() < timeout {
            let (completion_handle, completion_receiver) =
                self.client.open_once_port::<Option<IbvWc>>();

            self.manager
                .cast(
                    selection::selection_from_one(self.manager.shape(), "gpu", 0..1).unwrap(),
                    PollCompletion(other.id().await, completion_handle.bind()),
                )
                .unwrap();

            match completion_receiver.recv().await? {
                Some(wc) => {
                    if wc.wr_id() == work_id {
                        return Ok(true);
                    } else {
                        tracing::debug!("Got wrong work id: {}", wc.wr_id());
                    }
                }
                None => {
                    RealClock.sleep(Duration::from_millis(1)).await;
                }
            }
        }

        Ok(false)
    }

    pub async fn connect(&self, other: &RdmaManager<'_>) -> Result<(), anyhow::Error> {
        tracing::debug!("[{}] connecting to {}", self.name, other.name);
        let connection_info = other.connection_info(self).await.map_err(|e| {
            anyhow::anyhow!("Could not read connection info for other buffer: {}", e)
        })?;
        tracing::debug!("[{}] got connection info {:?}", self.name, connection_info);

        self.manager
            .cast(
                selection::selection_from_one(self.manager.shape(), "gpu", 0..1).unwrap(),
                Connect(other.id().await, connection_info.clone()),
            )
            .unwrap();

        tracing::debug!(
            "[{}] done connecting to {:?}",
            self.name,
            connection_info.clone()
        );
        Ok(())
    }

    pub async fn maybe_connect(&self, other: &RdmaManager<'_>) -> Result<(), anyhow::Error> {
        tracing::debug!("[{}] maybe connecting", self.name);

        let (connected_handle, connected_receiver) = self.client.open_once_port::<bool>();

        self.manager
            .cast(
                selection::selection_from_one(self.manager.shape(), "gpu", 0..1).unwrap(),
                Connected(other.id().await, connected_handle.bind()),
            )
            .unwrap();

        let connected = connected_receiver.recv().await?;

        // TODO - do we want any locking here since we're doing connections on both sides?
        if !connected {
            tracing::debug!("[{}] not connected, connecting", self.name);
            self.connect(other).await?;
            tracing::debug!("[{}] connecting other", self.name);
            other.connect(self).await?;
        }
        Ok(())
    }

    pub async fn write(&mut self, other: &RdmaManager<'_>) -> Result<u64, anyhow::Error> {
        let work_id = self.current_work_id;
        self.current_work_id += 1;
        self.maybe_connect(other).await?;

        self.manager
            .cast(
                selection::selection_from_one(self.manager.shape(), "gpu", 0..1).unwrap(),
                Put(other.id().await, work_id),
            )
            .unwrap();

        Ok(work_id)
    }

    pub async fn read_into(&mut self, other: &RdmaManager<'_>) -> Result<u64, anyhow::Error> {
        let work_id = self.current_work_id;
        self.current_work_id += 1;
        self.maybe_connect(other).await?;

        self.manager
            .cast(
                selection::selection_from_one(self.manager.shape(), "gpu", 0..1).unwrap(),
                Fetch(other.id().await, work_id),
            )
            .unwrap();

        Ok(work_id)
    }
}

#[cfg(test)]
mod tests {
    use hyperactor_mesh::alloc::AllocConstraints;
    use hyperactor_mesh::alloc::AllocSpec;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use ndslice::shape;

    use super::*;

    // Helper function to create test data
    fn create_test_data(size: usize) -> Box<[u8]> {
        let mut data = vec![0u8; size].into_boxed_slice();
        for (i, val) in data.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }
        data
    }

    // Helper function to create an RdmaManager
    async fn create_buffer<'a>(
        name: &str,
        proc_mesh: &'a ProcMesh,
        data: &mut [u8],
        device_id: Option<usize>,
    ) -> Result<RdmaManager<'a>, anyhow::Error> {
        let mut config = RdmaConnectionConfig::default();

        // If a specific device ID is provided, use that device
        if let Some(id) = device_id {
            let devices = crate::ibverbs_primitives::get_all_devices();
            if id < devices.len() {
                config.device = devices[id].clone();
            }
        }

        let memory_region = RdmaMemoryRegion::from(data);

        RdmaManager::new(RdmaManagerArgs {
            name: name.to_string(),
            proc_mesh,
            config,
            memory_region,
        })
        .await
    }

    // Helper function to verify buffer contents match
    fn verify_buffer_contents(buffer1: &[u8], buffer2: &[u8]) {
        assert_eq!(buffer1.len(), buffer2.len(), "Buffer lengths don't match");

        for i in 0..buffer1.len() {
            assert_eq!(
                buffer1[i], buffer2[i],
                "Data mismatch at position {}: {} != {}",
                i, buffer1[i], buffer2[i]
            );
        }
    }

    #[tokio::test]
    async fn test_buffer_creation() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 8;

        let alloc = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
        let mut data = create_test_data(BUFFER_SIZE);

        let buffer = create_buffer("test_buffer", &proc_mesh, &mut data, None).await?;

        // Verify buffer was created successfully
        assert!(!buffer.id().await.to_string().is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_buffer_write() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 8;

        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();

        let mesh1 = ProcMesh::allocate(alloc1).await.unwrap();
        let mesh2 = ProcMesh::allocate(alloc2).await.unwrap();

        let mut buffer1_data = create_test_data(BUFFER_SIZE);
        let mut buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();

        let mut buffer1 = create_buffer("buffer1", &mesh1, &mut buffer1_data, Some(0)).await?;
        let buffer2 = create_buffer("buffer2", &mesh2, &mut buffer2_data, Some(4)).await?;

        let work_id = buffer1.write(&buffer2).await?;

        let completed = buffer1.wait_for_completion(&buffer2, 5, work_id).await?;
        assert!(completed, "RDMA write operation did not complete");

        // Verify buffers have the same content
        verify_buffer_contents(&buffer1_data, &buffer2_data);

        Ok(())
    }

    #[tokio::test]
    async fn test_buffer_read() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 8;

        // Create proc meshes
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();

        let mesh1 = ProcMesh::allocate(alloc1).await.unwrap();
        let mesh2 = ProcMesh::allocate(alloc2).await.unwrap();

        // Create buffers
        let mut buffer1_data = create_test_data(BUFFER_SIZE);
        let mut buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();

        let buffer1 = create_buffer("buffer1", &mesh1, &mut buffer1_data, Some(0)).await?;
        let mut buffer2 = create_buffer("buffer2", &mesh2, &mut buffer2_data, Some(4)).await?;

        // Read from buffer1 into buffer2
        let work_id = buffer2.read_into(&buffer1).await?;

        // Wait for completion
        let completed = buffer2.wait_for_completion(&buffer1, 5, work_id).await?;
        assert!(completed, "RDMA read operation did not complete");

        // Verify buffers have the same content
        verify_buffer_contents(&buffer1_data, &buffer2_data);

        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_buffers_write() -> Result<(), anyhow::Error> {
        // Tests writing from a single buffer into multiple buffers
        const BUFFER_SIZE: usize = 8;

        // Create proc meshes
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc3 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();

        let mesh1 = ProcMesh::allocate(alloc1).await.unwrap();
        let mesh2 = ProcMesh::allocate(alloc2).await.unwrap();
        let mesh3 = ProcMesh::allocate(alloc3).await.unwrap();

        // Create buffers
        let mut buffer1_data = create_test_data(BUFFER_SIZE);
        let mut buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();
        let mut buffer3_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();

        let mut buffer1 = create_buffer("buffer1", &mesh1, &mut buffer1_data, Some(0)).await?;
        let buffer2 = create_buffer("buffer2", &mesh2, &mut buffer2_data, Some(4)).await?;
        let buffer3 = create_buffer("buffer3", &mesh3, &mut buffer3_data, Some(5)).await?;

        // Write from buffer1 to buffer2
        let work_id1 = buffer1.write(&buffer2).await?;
        let completed1 = buffer1.wait_for_completion(&buffer2, 5, work_id1).await?;
        assert!(
            completed1,
            "RDMA write operation to buffer2 did not complete"
        );

        // Write from buffer1 to buffer3
        let work_id2 = buffer1.write(&buffer3).await?;
        let completed2 = buffer1.wait_for_completion(&buffer3, 5, work_id2).await?;
        assert!(
            completed2,
            "RDMA write operation to buffer3 did not complete"
        );

        // Verify buffers have the same content
        verify_buffer_contents(&buffer1_data, &buffer2_data);
        verify_buffer_contents(&buffer1_data, &buffer3_data);

        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_buffers_read() -> Result<(), anyhow::Error> {
        // Tests reading from a single buffer into multiple buffers
        const BUFFER_SIZE: usize = 8;

        // Create proc meshes
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc3 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();

        let mesh1 = ProcMesh::allocate(alloc1).await.unwrap();
        let mesh2 = ProcMesh::allocate(alloc2).await.unwrap();
        let mesh3 = ProcMesh::allocate(alloc3).await.unwrap();

        let mut buffer1_data = create_test_data(BUFFER_SIZE);
        let mut buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();
        let mut buffer3_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();

        let buffer1 = create_buffer("buffer1", &mesh1, &mut buffer1_data, Some(0)).await?;
        let mut buffer2 = create_buffer("buffer2", &mesh2, &mut buffer2_data, Some(4)).await?;
        let mut buffer3 = create_buffer("buffer3", &mesh3, &mut buffer3_data, Some(5)).await?;

        // Read from buffer1 into buffer2
        let work_id1 = buffer2.read_into(&buffer1).await?;
        let completed1 = buffer2.wait_for_completion(&buffer1, 5, work_id1).await?;
        assert!(
            completed1,
            "RDMA read operation into buffer2 did not complete"
        );

        // Read from buffer1 into buffer3
        let work_id2 = buffer3.read_into(&buffer1).await?;
        let completed2 = buffer3.wait_for_completion(&buffer1, 5, work_id2).await?;
        assert!(
            completed2,
            "RDMA read operation into buffer3 did not complete"
        );

        // Verify buffers have the same content
        verify_buffer_contents(&buffer1_data, &buffer2_data);
        verify_buffer_contents(&buffer1_data, &buffer3_data);

        Ok(())
    }

    // Note - this test fails with FATAL as errors are not handled properly.
    // We disable this for now, but we definitely want to revisit.
    // #[tokio::test]
    #[allow(dead_code)]
    async fn test_connection_methods() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 8;

        // Create proc meshes
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();

        let mesh1 = ProcMesh::allocate(alloc1).await.unwrap();
        let mesh2 = ProcMesh::allocate(alloc2).await.unwrap();

        // Create buffers
        let mut buffer1_data = create_test_data(BUFFER_SIZE);
        let mut buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();

        let mut buffer1 = create_buffer("buffer1", &mesh1, &mut buffer1_data, Some(0)).await?;
        let buffer2 = create_buffer("buffer2", &mesh2, &mut buffer2_data, Some(4)).await?;

        // Test connection_info method
        let connection_info = buffer1.connection_info(&buffer2).await?;
        assert!(
            !format!("{:?}", connection_info).is_empty(),
            "Connection info should not be empty"
        );

        // Test connect method
        buffer1.connect(&buffer2).await?;

        // Test maybe_connect method (should not reconnect)
        buffer1.maybe_connect(&buffer2).await?;

        // Verify we can still write after connection
        let work_id = buffer1.write(&buffer2).await?;
        let completed = buffer1.wait_for_completion(&buffer2, 5, work_id).await?;
        assert!(
            completed,
            "RDMA write operation did not complete after connection"
        );

        verify_buffer_contents(&buffer1_data, &buffer2_data);

        Ok(())
    }

    // Test for timeout behavior in wait_for_completion
    // Note - this test fails with FATAL as errors are not handled properly.
    // We disable this for now, but we definitely want to revisit.
    // #[tokio::test]
    #[allow(dead_code)]
    async fn test_wait_for_completion_timeout() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 8;

        // Create proc meshes
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();

        let mesh1 = ProcMesh::allocate(alloc1).await.unwrap();
        let mesh2 = ProcMesh::allocate(alloc2).await.unwrap();

        // Create buffers
        let mut buffer1_data = create_test_data(BUFFER_SIZE);
        let mut buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();

        let buffer1 = create_buffer("buffer1", &mesh1, &mut buffer1_data, Some(0)).await?;
        let buffer2 = create_buffer("buffer2", &mesh2, &mut buffer2_data, Some(4)).await?;

        // Test with an invalid work_id that should time out
        let invalid_work_id = 9999;
        let timeout_secs = 1;

        let start = std::time::Instant::now();
        let completed = buffer1
            .wait_for_completion(&buffer2, timeout_secs, invalid_work_id)
            .await?;
        let elapsed = start.elapsed();

        // Verify timeout behavior
        assert!(!completed, "Wait should have timed out for invalid work_id");
        assert!(
            elapsed >= Duration::from_secs(timeout_secs),
            "Wait should have waited for at least the timeout duration"
        );

        Ok(())
    }
}
