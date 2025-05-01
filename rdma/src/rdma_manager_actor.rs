//! # RDMA Manager
//!
//! This module defines `RdmaManagerActor`, which is responsible for managing RDMA connections
//! and operations. It uses `hyperactor` to handle long-lived connections,
//! asynchronous messsage passing and actor lifecycle management.
//!
//! In the context of a Monarch workload, `RdmaManagerActor` is meant to run on a single host.
//! `RdmaManagerActor` is meant only to interact with other `RdmaManagerActor`s in the same Monarch
//! cluster which may or may not be on the same host.
//!
//! ## Usage
//!
//! For complete usage examples, see the test module at the bottom of this file,
//! particularly `test_rdma_write_loopback` and `test_rdma_read_loopback`.
use std::collections::HashMap;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::OncePortRef;
use hyperactor::supervision::ActorSupervisionEvent;
use hyperactor_mesh::actor_mesh::Cast;
use serde::Deserialize;
use serde::Serialize;

use crate::ibverbs_primitives::IbvWc;
use crate::ibverbs_primitives::RdmaConnectionInfo;
use crate::ibverbs_primitives::RdmaMemoryRegion;
use crate::ibverbs_primitives::RdmaOperation;
use crate::rdma_connection::RdmaConnectionConfig;
use crate::rdma_connection::RdmaConnectionPoint;

/// `RdmaManagerActor` is responsible for managing RDMA connections and operations within a Monarch cluster.
/// It is an actor that can handle various messages to register memory regions, connect to other
/// `RdmaManager` instances, and perform RDMA operations such as reading from and writing to remote
/// memory regions. The manager maintains a map of connections to other actors and can initialize
/// connections as needed.
///
/// Any RDMA operation requires a pair, and therefore all commands here assume the presence
/// of another RdmaManagerActor, represented by its actor_id.
#[derive(Debug)]
#[hyperactor::export_spawn(Cast<Register>, Cast<Connected>, Cast<Connect>, Cast<ConnectionInfo>, Cast<Fetch>, Cast<Put>, Cast<PollCompletion>)]
pub struct RdmaManagerActor {
    connection_map: HashMap<ActorId, RdmaConnectionPoint>,
    config: Option<RdmaConnectionConfig>,
    buffer: Option<RdmaMemoryRegion>,
}

#[async_trait]
impl Actor for RdmaManagerActor {
    type Params = ();

    async fn new(_params: Self::Params) -> Result<Self, anyhow::Error> {
        Ok(Self {
            connection_map: HashMap::new(),
            config: None,
            buffer: None,
        })
    }

    async fn handle_supervision_event(
        &mut self,
        _this: &Instance<Self>,
        _event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        // We likely want to bubble this up?
        tracing::info!("RdmaManagerActor supervision event: {:?}", _event);
        tracing::info!("RdmaManagerActor error occurred, stop the worker process, exit code: 1");
        std::process::exit(1);
    }
}

// Actor message types
#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Register(pub RdmaConnectionConfig, pub RdmaMemoryRegion);

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Connected(pub ActorId, pub OncePortRef<bool>);

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Connect(pub ActorId, pub RdmaConnectionInfo);

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Fetch(pub ActorId, pub u64);

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Put(pub ActorId, pub u64);

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct PollCompletion(pub ActorId, pub OncePortRef<Option<IbvWc>>);

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct ConnectionInfo(pub ActorId, pub OncePortRef<RdmaConnectionInfo>);

impl RdmaManagerActor {
    /// Convenience utility to create a new RdmaConnection.
    pub async fn initialize_connection(&mut self, other: ActorId) -> Result<(), anyhow::Error> {
        if let std::collections::hash_map::Entry::Vacant(e) =
            self.connection_map.entry(other.clone())
        {
            let connection = RdmaConnectionPoint::new(
                self.config.as_ref().unwrap().clone(),
                self.buffer.as_ref().unwrap().clone(),
            )
            .map_err(|e| anyhow::anyhow!("Could not create RdmaConnection: {}", e))?;
            // Store the uninitialized connection in the map
            e.insert(connection);
        }
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<Register>> for RdmaManagerActor {
    /// Register a memory region and configuration.
    async fn handle(
        &mut self,
        _this: &Instance<Self>,
        Cast {
            rank: _,
            message: Register(config, buffer),
            ..
        }: Cast<Register>,
    ) -> Result<(), anyhow::Error> {
        self.config = Some(config);
        self.buffer = Some(buffer);
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<Connected>> for RdmaManagerActor {
    /// Poll whether or not this RdmaManager has connected to the other.
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        Cast {
            rank: _,
            message: Connected(other, reply),
            ..
        }: Cast<Connected>,
    ) -> Result<(), anyhow::Error> {
        let connected = self.connection_map.contains_key(&other);
        let _send_result = reply.send(this, connected);
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<Connect>> for RdmaManagerActor {
    /// Connect this RdmaManager to the other, i.e. complete the handshake and transition
    /// ibv state.
    async fn handle(
        &mut self,
        _this: &Instance<Self>,
        Cast {
            rank: _,
            message: Connect(other, endpoint),
            ..
        }: Cast<Connect>,
    ) -> Result<(), anyhow::Error> {
        if !self.connection_map.contains_key(&other.clone()) {
            self.initialize_connection(other.clone()).await?;
        }
        let connection = self
            .connection_map
            .get_mut(&other)
            .ok_or_else(|| anyhow::anyhow!("No connection found for actor {}", other))?;
        connection
            .connect(&endpoint)
            .map_err(|e| anyhow::anyhow!("Could not connect to RDMA endpoint: {}", e))?;
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<Fetch>> for RdmaManagerActor {
    /// Read from the partner's RDMA buffer into this.
    async fn handle(
        &mut self,
        _this: &Instance<Self>,
        Cast {
            rank: _,
            message: Fetch(other, work_id),
            ..
        }: Cast<Fetch>,
    ) -> Result<(), anyhow::Error> {
        let connection = self
            .connection_map
            .get_mut(&other)
            .ok_or_else(|| anyhow::anyhow!("No connection found for actor {}", other))?;
        connection
            .post_send(
                ..self.buffer.as_ref().unwrap().len(),
                work_id,
                true,
                RdmaOperation::Read,
            )
            .map_err(|e| anyhow::anyhow!("Could not post RDMA read: {}", e))?;
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<Put>> for RdmaManagerActor {
    /// Write this RDMA buffer into the partner's.
    async fn handle(
        &mut self,
        _this: &Instance<Self>,
        Cast {
            rank: _,
            message: Put(other, work_id),
            ..
        }: Cast<Put>,
    ) -> Result<(), anyhow::Error> {
        let connection = self
            .connection_map
            .get_mut(&other)
            .ok_or_else(|| anyhow::anyhow!("No connection found for actor {}", other))?;
        connection
            .post_send(
                ..self.buffer.as_ref().unwrap().len(),
                work_id,
                true,
                RdmaOperation::Write,
            )
            .map_err(|e| anyhow::anyhow!("Could not post RDMA write: {}", e))?;
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<PollCompletion>> for RdmaManagerActor {
    /// Check whether or not a given work request has completed.
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        Cast {
            rank: _,
            message: PollCompletion(other, reply),
            ..
        }: Cast<PollCompletion>,
    ) -> Result<(), anyhow::Error> {
        let connection = self
            .connection_map
            .get_mut(&other)
            .ok_or_else(|| anyhow::anyhow!("No connection found for actor {}", other))?;

        let wc = connection
            .poll_completion()
            .map_err(|e| anyhow::anyhow!("Could not poll completion: {}", e))?;
        let _send_result = reply.send(this, wc);
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<ConnectionInfo>> for RdmaManagerActor {
    /// Get the partner-associated connection info, creating one if needed.
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        Cast {
            rank: _,
            message: ConnectionInfo(other, reply),
            ..
        }: Cast<ConnectionInfo>,
    ) -> Result<(), anyhow::Error> {
        if !self.connection_map.contains_key(&other.clone()) {
            self.initialize_connection(other.clone()).await?;
        }
        let connection_info = self
            .connection_map
            .get(&other)
            .unwrap()
            .get_connection_info()?;
        let _send_result = reply.send(this, connection_info);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
    use hyperactor_mesh::ActorMesh;
    use hyperactor_mesh::Mesh;
    use hyperactor_mesh::ProcMesh;
    use hyperactor_mesh::alloc::AllocConstraints;
    use hyperactor_mesh::alloc::AllocSpec;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use ndslice::selection;
    use ndslice::shape;
    use tokio::time::Duration;

    use super::*;
    use crate::ibverbs_primitives;

    struct RdmaManagerTestEnv {
        buffer1: Box<[u8]>,
        buffer2: Box<[u8]>,
        // Note: Using a static `proc_mesh` is suboptimal because it leaks memory,
        // but it is necessary for the lifetime of `actor_mesh` in this test setup.
        // This should be fine for the sake of testing though.
        proc_mesh: &'static ProcMesh,
        actor_mesh: ActorMesh<'static, RdmaManagerActor>,
        id1: ActorId,
        id2: ActorId,
    }

    impl RdmaManagerTestEnv {
        async fn setup(buffer_size: usize) -> Result<Self, anyhow::Error> {
            let alloc = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { replica = 1, host = 1, gpu = 2 },
                    constraints: AllocConstraints::none(),
                })
                .await
                .unwrap();

            let proc_mesh = Box::leak(Box::new(ProcMesh::allocate(alloc).await.unwrap()));
            let actor_mesh = proc_mesh.spawn("rdma_manager", &()).await.unwrap();

            let binding = actor_mesh.get(0).unwrap();
            let id1 = binding.actor_id();

            let binding = actor_mesh.get(1).unwrap();
            let id2 = binding.actor_id();

            let mut buffer1 = vec![0u8; buffer_size].into_boxed_slice();
            let buffer2 = vec![0u8; buffer_size].into_boxed_slice();

            // Fill buffer1 with test data
            for (i, val) in buffer1.iter_mut().enumerate() {
                *val = (i % 256) as u8;
            }

            Ok(Self {
                buffer1,
                buffer2,
                proc_mesh,
                actor_mesh,
                id1: id1.clone(),
                id2: id2.clone(),
            })
        }

        async fn initialize(
            &mut self,
            devices: Option<(usize, usize)>,
        ) -> Result<(), anyhow::Error> {
            let (config1, config2) = if let Some((dev1_idx, dev2_idx)) = devices {
                let all_devices = ibverbs_primitives::get_all_devices();
                if all_devices.len() < 2 {
                    return Err(anyhow::anyhow!(
                        "Need at least 2 RDMA devices for this test"
                    ));
                }

                (
                    RdmaConnectionConfig {
                        device: all_devices.clone().into_iter().nth(dev1_idx).unwrap(),
                        ..Default::default()
                    },
                    RdmaConnectionConfig {
                        device: all_devices.clone().into_iter().nth(dev2_idx).unwrap(),
                        ..Default::default()
                    },
                )
            } else {
                (
                    RdmaConnectionConfig::default(),
                    RdmaConnectionConfig::default(),
                )
            };

            let region1 = RdmaMemoryRegion::from(&mut self.buffer1[..]);
            let region2 = RdmaMemoryRegion::from(&mut self.buffer2[..]);

            self.actor_mesh
                .cast(
                    selection::selection_from_one(self.actor_mesh.shape(), "gpu", 0..1).unwrap(),
                    Register(config1, region1),
                )
                .unwrap();
            self.actor_mesh
                .cast(
                    selection::selection_from_one(self.actor_mesh.shape(), "gpu", 1..2).unwrap(),
                    Register(config2, region2),
                )
                .unwrap();

            // Get the endpoints
            let (endpoint_handle, endpoint_receiver) = self
                .proc_mesh
                .client()
                .open_once_port::<RdmaConnectionInfo>();

            self.actor_mesh
                .cast(
                    selection::selection_from_one(self.actor_mesh.shape(), "gpu", 0..1).unwrap(),
                    ConnectionInfo(self.id2.clone(), endpoint_handle.bind()),
                )
                .unwrap();
            let endpoint1 = endpoint_receiver.recv().await.unwrap();

            let (endpoint_handle, endpoint_receiver) = self
                .proc_mesh
                .client()
                .open_once_port::<RdmaConnectionInfo>();

            self.actor_mesh
                .cast(
                    selection::selection_from_one(self.actor_mesh.shape(), "gpu", 1..2).unwrap(),
                    ConnectionInfo(self.id1.clone(), endpoint_handle.bind()),
                )
                .unwrap();
            let endpoint2 = endpoint_receiver.recv().await.unwrap();

            // Connect to endpoints
            self.actor_mesh
                .cast(
                    selection::selection_from_one(self.actor_mesh.shape(), "gpu", 0..1).unwrap(),
                    Connect(self.id2.clone(), endpoint2),
                )
                .unwrap();
            self.actor_mesh
                .cast(
                    selection::selection_from_one(self.actor_mesh.shape(), "gpu", 1..2).unwrap(),
                    Connect(self.id1.clone(), endpoint1),
                )
                .unwrap();

            Ok(())
        }

        async fn wait_for_completion(
            &self,
            actor_idx: usize,
            other_id: ActorId,
            wr_id: u64,
            timeout_secs: u64,
        ) -> Result<bool, anyhow::Error> {
            let timeout = Duration::from_secs(timeout_secs);
            let start_time = std::time::Instant::now();

            while start_time.elapsed() < timeout {
                let (completion_handle, completion_receiver) =
                    self.proc_mesh.client().open_once_port::<Option<IbvWc>>();

                self.actor_mesh
                    .cast(
                        selection::selection_from_one(
                            self.actor_mesh.shape(),
                            "gpu",
                            actor_idx..(actor_idx + 1),
                        )
                        .unwrap(),
                        PollCompletion(other_id.clone(), completion_handle.bind()),
                    )
                    .unwrap();

                let wc = completion_receiver.recv().await.unwrap();
                match wc {
                    Some(wc) => {
                        if wc.wr_id() == wr_id {
                            return Ok(true);
                        }
                    }
                    None => {
                        RealClock.sleep(Duration::from_millis(1)).await;
                    }
                }
            }

            Ok(false)
        }

        async fn verify_buffers(&self, size: usize) -> Result<(), anyhow::Error> {
            for i in 0..size {
                assert_eq!(
                    self.buffer1[i], self.buffer2[i],
                    "Data mismatch at position {}: {} != {}",
                    i, self.buffer1[i], self.buffer2[i]
                );
            }

            Ok(())
        }
    }

    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_write_loopback() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let mut env = RdmaManagerTestEnv::setup(BSIZE).await?;
        env.initialize(None).await?;

        env.actor_mesh
            .cast(
                selection::selection_from_one(env.actor_mesh.shape(), "gpu", 0..1).unwrap(),
                Put(env.id2.clone(), 0),
            )
            .unwrap();

        let completed = env.wait_for_completion(0, env.id2.clone(), 0, 1).await?;
        assert!(completed, "RDMA write operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_read_loopback() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let mut env = RdmaManagerTestEnv::setup(BSIZE).await?;
        env.initialize(None).await?;

        env.actor_mesh
            .cast(
                selection::selection_from_one(env.actor_mesh.shape(), "gpu", 1..2).unwrap(),
                Fetch(env.id1.clone(), 0),
            )
            .unwrap();

        let completed = env.wait_for_completion(1, env.id1.clone(), 0, 1).await?;
        assert!(completed, "RDMA read operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_read_separate_devices() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let mut env = RdmaManagerTestEnv::setup(BSIZE).await?;
        env.initialize(Some((0, 4))).await?;

        env.actor_mesh
            .cast(
                selection::selection_from_one(env.actor_mesh.shape(), "gpu", 1..2).unwrap(),
                Fetch(env.id1.clone(), 0),
            )
            .unwrap();

        let completed = env.wait_for_completion(1, env.id1.clone(), 0, 1).await?;
        assert!(completed, "RDMA read operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_write_separate_devices() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let mut env = RdmaManagerTestEnv::setup(BSIZE).await?;
        env.initialize(Some((0, 4))).await?;

        env.actor_mesh
            .cast(
                selection::selection_from_one(env.actor_mesh.shape(), "gpu", 0..1).unwrap(),
                Put(env.id2.clone(), 0),
            )
            .unwrap();

        let completed = env.wait_for_completion(0, env.id2.clone(), 0, 1).await?;
        assert!(completed, "RDMA write operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }
}
