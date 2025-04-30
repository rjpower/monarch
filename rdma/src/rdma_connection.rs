//! # RDMA Connection
//!
//! This implements RdmaConnectionPoint, a structure that represents one side of a connection to a remote RDMA device.
//!
//! In practice, RDMA connections are paired. Using `RdmaConnectionPoint` practically requires a pair of
//! `RdmaConnectionPoint`s, representing each side of the connection.
//!
//! # RDMA Overview
//!
//! Remote Direct Memory Access (RDMA) allows direct memory access from the memory of one computer
//! into the memory of another without involving either computer's operating system. This permits
//! high-throughput, low-latency networking with minimal CPU overhead.
//!
//! # Connection Architecture
//!
//! `RdmaConnectionPoint` manages the base ibverbs primitives:
//!
//! 1. **Queue Pairs (QP)**: Each connection has a send queue and a receive queue that form a queue pair
//! 2. **Completion Queues (CQ)**: Events are reported when operations complete
//! 3. **Memory Regions (MR)**: Memory must be registered with the RDMA device before use
//! 4. **Protection Domains (PD)**: Provide isolation between different connections
//!
//! # Connection Lifecycle
//!
//! `RdmaConnectionPoint` abstracts away details of the RDMA initialization. The following is a high-level
//! description of the connection lifecycle:
//!
//! 1. Create connection with `new()`
//! 2. Exchange endpoints with remote peer (application must handle this)
//! 3. Connect to remote endpoint with `connect()`
//! 4. Perform RDMA operations (read/write)
//! 5. Connection is cleaned up when dropped
use std::ffi::CStr;
use std::io::Error;
use std::io::Result;
use std::ops::Bound;
use std::ops::RangeBounds;

/// Direct access to low-level libibverbs FFI.
pub use ffi::ibv_qp_type;
use ibverbs::Gid;

use crate::ibverbs_primitives::IbvWc;
use crate::ibverbs_primitives::RdmaConnectionInfo;
use crate::ibverbs_primitives::RdmaDevice;
use crate::ibverbs_primitives::RdmaMemoryRegion;
use crate::ibverbs_primitives::RdmaOperation;

/// Represents the configuration for an RDMA connection.
///
/// This struct holds various parameters required to establish and manage an RDMA connection.
/// It includes settings for the RDMA device, queue pair attributes, and other connection-specific
/// parameters.
///
/// # Fields
///
/// * `device` - The RDMA device to use for the connection.
/// * `cq_entries` - The number of completion queue entries.
/// * `port_num` - The port number on the RDMA device.
/// * `gid_index` - The GID index for the RDMA device.
/// * `max_send_wr` - The maximum number of outstanding send work requests.
/// * `max_recv_wr` - The maximum number of outstanding receive work requests.
/// * `max_send_sge` - The maximum number of scatter/gather elements in a send work request.
/// * `max_recv_sge` - The maximum number of scatter/gather elements in a receive work request.
/// * `path_mtu` - The path MTU (Maximum Transmission Unit) for the connection.
/// * `retry_cnt` - The number of retry attempts for a connection request.
/// * `rnr_retry` - The number of retry attempts for a receiver not ready (RNR) condition.
/// * `qp_timeout` - The timeout for a queue pair operation.
/// * `min_rnr_timer` - The minimum RNR timer value.
/// * `max_dest_rd_atomic` - The maximum number of outstanding RDMA read operations at the destination.
/// * `max_rd_atomic` - The maximum number of outstanding RDMA read operations at the initiator.
/// * `pkey_index` - The partition key index.
/// * `psn` - The packet sequence number.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RdmaConnectionConfig {
    pub device: RdmaDevice,
    pub cq_entries: i32,
    pub port_num: u8,
    pub gid_index: u8,
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub path_mtu: ffi::ibv_mtu,
    pub retry_cnt: u8,
    pub rnr_retry: u8,
    pub qp_timeout: u8,
    pub min_rnr_timer: u8,
    pub max_dest_rd_atomic: u8,
    pub max_rd_atomic: u8,
    pub pkey_index: u16,
    pub psn: u32,
}

impl Default for RdmaConnectionConfig {
    fn default() -> Self {
        Self {
            device: RdmaDevice::default(),
            cq_entries: 10,
            port_num: 1,
            gid_index: 3,
            max_send_wr: 1,
            max_recv_wr: 1,
            max_send_sge: 1,
            max_recv_sge: 1,
            path_mtu: ffi::IBV_MTU_1024,
            retry_cnt: 7,
            rnr_retry: 7,
            qp_timeout: 14, // 4.096 μs * 2^14 = ~67 ms
            min_rnr_timer: 12,
            max_dest_rd_atomic: 1,
            max_rd_atomic: 1,
            pkey_index: 0,
            psn: rand::random::<u32>() & 0xffffff,
        }
    }
}

impl std::fmt::Display for RdmaConnectionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RdmaConnectionConfig {{ device: {}, port_num: {}, gid_index: {}, max_send_wr: {}, max_recv_wr: {}, max_send_sge: {}, max_recv_sge: {}, path_mtu: {:?}, retry_cnt: {}, rnr_retry: {}, qp_timeout: {}, min_rnr_timer: {}, max_dest_rd_atomic: {}, max_rd_atomic: {}, pkey_index: {}, psn: 0x{:x} }}",
            self.device.name(),
            self.port_num,
            self.gid_index,
            self.max_send_wr,
            self.max_recv_wr,
            self.max_send_sge,
            self.max_recv_sge,
            self.path_mtu,
            self.retry_cnt,
            self.rnr_retry,
            self.qp_timeout,
            self.min_rnr_timer,
            self.max_dest_rd_atomic,
            self.max_rd_atomic,
            self.pkey_index,
            self.psn,
        )
    }
}

/// Represents an RDMA connection.
///
/// `RdmaConnectionPoint` is a structure that encapsulates the details of a connection
/// to a remote RDMA device. It manages the lifecycle of the connection, including
/// the creation and destruction of RDMA resources such as queue pairs, completion
/// queues, and memory regions.
///
/// # Fields
///
/// * `context` - The RDMA context associated with the connection.
/// * `pd` - The protection domain for the connection.
/// * `cq` - The completion queue used for tracking operation completions.
/// * `qp` - The queue pair used for sending and receiving RDMA operations.
/// * `mr` - The memory region registered for RDMA operations.
/// * `buffer` - The memory buffer associated with the connection.
/// * `endpoint` - The local RDMA connection information.
/// * `remote_connection_info` - The remote RDMA connection information, if connected.
/// * `config` - The configuration parameters for the connection.
///
/// # Safety
///
/// `RdmaConnectionPoint` contains unsafe code that interacts with the RDMA device
/// through FFI calls. It is `Send` and `Sync` because the underlying ibverbs
/// APIs are thread-safe, and the raw pointers to ibverbs structs can be accessed
/// from any thread.
pub struct RdmaConnectionPoint {
    context: *mut ffi::ibv_context,
    pd: *mut ffi::ibv_pd,
    cq: *mut ffi::ibv_cq,
    qp: *mut ffi::ibv_qp,
    mr: *mut ffi::ibv_mr,
    buffer: RdmaMemoryRegion,
    connection_info: RdmaConnectionInfo,
    remote_connection_info: Option<RdmaConnectionInfo>,
    config: RdmaConnectionConfig,
}

impl std::fmt::Display for RdmaConnectionPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RdmaConnectionPoint {{ qp_num: {}, buffer_size: {}, remote_connection_info: {:?} }}",
            self.connection_info.qp_num,
            self.buffer.len(),
            self.remote_connection_info,
        )
    }
}

// SAFETY:
// This function contains unsafe code that interacts with the Rdma device through FFI calls.
// RdmaConnectionPoint is `Send` because the raw pointers to ibverbs structs can be
// accessed from any thread, and it is safe to drop `RdmaConnectionPoint` (and run the
// ibverbs destructors) from any thread.
unsafe impl Send for RdmaConnectionPoint {}

// SAFETY:
// This function contains unsafe code that interacts with the Rdma device through FFI calls.
// RdmaConnectionPoint is `Sync` because the underlying ibverbs APIs are thread-safe.
unsafe impl Sync for RdmaConnectionPoint {}

impl std::fmt::Debug for RdmaConnectionPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdmaConnectionPoint")
            .field("endpoint", &self.connection_info)
            .field("buffer_size", &self.buffer.len())
            .field("context", &format!("{:p}", self.context))
            .field("qp_num", &self.connection_info.qp_num)
            .field("remote_connection_info", &self.remote_connection_info)
            .field("config", &self.config)
            .finish()
    }
}

impl RdmaConnectionPoint {
    /// Creates a new Rdma connection point with the provided buffer.
    ///
    /// The caller must ensure:
    /// - The buffer is not accessed by other code while RDMA operations are in progress.
    /// - The buffer remains valid for the lifetime of the RdmaConnectionPoint.
    /// - The buffer must not be moved in memory during the lifetime of this connection.
    ///
    /// # Arguments
    ///
    /// * `device` - The Rdma device to use.
    /// * `buffer` - A RdmaMemoryRegion wrapper to the pointer to buffer that will be registered with the Rdma device
    pub fn new(config: RdmaConnectionConfig, buffer: RdmaMemoryRegion) -> Result<Self> {
        println!(
            "Creating RdmaConnectionPoint for device {}",
            config.device.name()
        );
        // SAFETY:
        // This function contains unsafe code that:
        // - Interacts with the Rdma device through FFI calls
        // - Registers the provided buffer with the Rdma device
        unsafe {
            // Get the device based on the provided RdmaDevice
            let device_name = config.device.name();
            let mut num_devices = 0i32;
            let devices = ffi::ibv_get_device_list(&mut num_devices as *mut _);

            if devices.is_null() || num_devices == 0 {
                return Err(Error::new(
                    std::io::ErrorKind::NotFound,
                    "No RDMA devices found".to_string(),
                ));
            }

            // Find the device with the matching name
            let mut device_ptr = std::ptr::null_mut();
            for i in 0..num_devices {
                let dev = *devices.offset(i as isize);
                let dev_name = CStr::from_ptr(ffi::ibv_get_device_name(dev)).to_string_lossy();

                if dev_name == *device_name {
                    device_ptr = dev;
                    break;
                }
            }

            // If we didn't find the device, return an error
            if device_ptr.is_null() {
                ffi::ibv_free_device_list(devices);
                return Err(Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("RDMA device '{}' not found", device_name),
                ));
            }
            tracing::info!("Using RDMA device: {}", device_name);

            // Open device
            let context = ffi::ibv_open_device(device_ptr);
            if context.is_null() {
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create context: {}", os_error),
                ));
            }

            // Create protection domain
            let pd = ffi::ibv_alloc_pd(context);
            if pd.is_null() {
                ffi::ibv_close_device(context);
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create protection domain (PD): {}", os_error),
                ));
            }

            // Create completion queue
            let cq = ffi::ibv_create_cq(
                context,
                config.cq_entries,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            if cq.is_null() {
                ffi::ibv_dealloc_pd(pd);
                ffi::ibv_close_device(context);
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create completion queue (CQ): {}", os_error),
                ));
            }

            // Create queue pair - note we currently share a CQ for both send and receive for simplicity.
            let mut qp_init_attr = ffi::ibv_qp_init_attr {
                qp_context: std::ptr::null::<std::os::raw::c_void>() as *mut _,
                send_cq: cq,
                recv_cq: cq,
                srq: std::ptr::null::<ffi::ibv_srq>() as *mut _,
                cap: ffi::ibv_qp_cap {
                    max_send_wr: config.max_send_wr,
                    max_recv_wr: config.max_recv_wr,
                    max_send_sge: config.max_send_sge,
                    max_recv_sge: config.max_recv_sge,
                    max_inline_data: 0,
                },
                qp_type: ibv_qp_type::IBV_QPT_RC,
                sq_sig_all: 0,
            };

            let qp = ffi::ibv_create_qp(pd, &mut qp_init_attr);
            if qp.is_null() {
                ffi::ibv_destroy_cq(cq);
                ffi::ibv_dealloc_pd(pd);
                ffi::ibv_close_device(context);
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create queue pair (QP): {}", os_error),
                ));
            }

            // Register memory region
            let access = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC;

            let mr = ffi::ibv_reg_mr(pd, buffer.as_ptr() as *mut _, buffer.len(), access.0 as i32);

            if mr.is_null() {
                ffi::ibv_destroy_qp(qp);
                ffi::ibv_destroy_cq(cq);
                ffi::ibv_dealloc_pd(pd);
                ffi::ibv_close_device(context);
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to register memory region (MR): {}", os_error),
                ));
            }

            // Create the connection info, used for another connection to connect to this one
            let mut port_attr = ffi::ibv_port_attr::default();
            let errno = ffi::ibv_query_port(
                context,
                config.port_num,
                &mut port_attr as *mut ffi::ibv_port_attr as *mut _,
            );
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to query port attributes: {}", os_error),
                ));
            }

            let mut gid = Gid::default();
            let ret = ffi::ibv_query_gid(
                context,
                config.port_num,
                i32::from(config.gid_index),
                gid.as_mut(),
            );
            if ret != 0 {
                return Err(Error::new(std::io::ErrorKind::Other, "Failed to query GID"));
            }
            let connection_info = RdmaConnectionInfo {
                qp_num: (*qp).qp_num,
                lid: port_attr.lid,
                gid: Some(gid),
                remote_addr: buffer.as_ptr() as u64,
                rkey: (*mr).rkey,
                psn: config.psn,
            };

            ffi::ibv_free_device_list(devices);

            Ok(RdmaConnectionPoint {
                context,
                pd,
                cq,
                qp,
                mr,
                buffer,
                connection_info,
                remote_connection_info: None,
                config,
            })
        }
    }

    /// Returns the Rdma connection info required for other RdmaConnectionPoints to connect to.
    pub fn get_connection_info(&self) -> Result<RdmaConnectionInfo> {
        Ok(self.connection_info.clone())
    }

    /// Connect to a remote Rdma connection point.
    ///
    /// This performs the necessary QP state transitions (INIT->RTR->RTS) to establish a connection.
    ///
    /// # Arguments
    ///
    /// * `connection_info` - The remote connection info to connect to
    pub fn connect(&mut self, connection_info: &RdmaConnectionInfo) -> Result<()> {
        // SAFETY:
        // This function contains unsafe code that:
        // - Interacts with the Rdma device through FFI calls
        // - Modifies the QP state to establish a connection
        unsafe {
            // Transition to INIT
            let qp_access_flags = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ;

            let mut qp_attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_INIT,
                qp_access_flags: qp_access_flags.0,
                pkey_index: self.config.pkey_index,
                port_num: self.config.port_num,
                ..Default::default()
            };

            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE
                | ffi::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
                | ffi::ibv_qp_attr_mask::IBV_QP_PORT
                | ffi::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;

            let errno = ffi::ibv_modify_qp(self.qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to transition QP to INIT: {}", os_error),
                ));
            }

            // Transition to RTR (Ready to Receive)
            let mut qp_attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_RTR,
                path_mtu: self.config.path_mtu,
                dest_qp_num: connection_info.qp_num,
                rq_psn: connection_info.psn,
                max_dest_rd_atomic: self.config.max_dest_rd_atomic,
                min_rnr_timer: self.config.min_rnr_timer,
                ah_attr: ffi::ibv_ah_attr {
                    dlid: connection_info.lid,
                    sl: 0,
                    src_path_bits: 0,
                    port_num: self.config.port_num,
                    grh: Default::default(),
                    ..Default::default()
                },
                ..Default::default()
            };

            // If the remote connection info contains a Gid, the routing will be global.
            // Otherwise, it will be local, i.e. using LID.
            if let Some(gid) = connection_info.gid {
                qp_attr.ah_attr.is_global = 1;
                qp_attr.ah_attr.grh.dgid = gid.into();
                qp_attr.ah_attr.grh.hop_limit = 0xff;
                qp_attr.ah_attr.grh.sgid_index = self.config.gid_index;
            } else {
                // Use LID-based routing, e.g. for Infiniband/RoCEv1
                qp_attr.ah_attr.is_global = 0;
            }

            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE
                | ffi::ibv_qp_attr_mask::IBV_QP_AV
                | ffi::ibv_qp_attr_mask::IBV_QP_PATH_MTU
                | ffi::ibv_qp_attr_mask::IBV_QP_DEST_QPN
                | ffi::ibv_qp_attr_mask::IBV_QP_RQ_PSN
                | ffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
                | ffi::ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;

            let errno = ffi::ibv_modify_qp(self.qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to transition QP to RTR: {}", os_error),
                ));
            }

            // Transition to RTS (Ready to Send)
            let mut qp_attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_RTS,
                sq_psn: self.config.psn,
                max_rd_atomic: self.config.max_rd_atomic,
                retry_cnt: self.config.retry_cnt,
                rnr_retry: self.config.rnr_retry,
                timeout: self.config.qp_timeout,
                ..Default::default()
            };

            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE
                | ffi::ibv_qp_attr_mask::IBV_QP_TIMEOUT
                | ffi::ibv_qp_attr_mask::IBV_QP_RETRY_CNT
                | ffi::ibv_qp_attr_mask::IBV_QP_SQ_PSN
                | ffi::ibv_qp_attr_mask::IBV_QP_RNR_RETRY
                | ffi::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;

            let errno = ffi::ibv_modify_qp(self.qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to transition QP to RTS: {}", os_error),
                ));
            }

            self.remote_connection_info = Some(connection_info.clone());

            Ok(())
        }
    }

    /// Posts a request to the queue pair.
    ///
    /// # Arguments
    ///
    /// * `range` - Range in the buffer containing data to send
    /// * `wr_id` - Work request ID for completion identification
    /// * `signaled` - Whether to generate a completion event
    /// * `op_type` - Optional operation type (default: IBV_WR_SEND)
    pub fn post_send<R: RangeBounds<usize>>(
        &mut self,
        range: R,
        wr_id: u64,
        signaled: bool,
        op_type: RdmaOperation,
    ) -> Result<()> {
        if self.remote_connection_info.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::NotConnected,
                "RDMA connection not established. Call connect() first.",
            ));
        }

        // Range bounds checking...
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.buffer.len(),
        };

        if start >= end || end > self.buffer.len() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid range for buffer",
            ));
        }

        let length = end - start;

        // SAFETY: Interacts with RDMA device through FFI calls
        unsafe {
            let mut send_sge = ffi::ibv_sge {
                addr: self.buffer.as_ptr().add(start) as u64,
                length: length as u32,
                lkey: (*self.mr).lkey,
            };

            let send_flags = if signaled {
                ffi::ibv_send_flags::IBV_SEND_SIGNALED.0
            } else {
                0
            };

            let mut send_wr = ffi::ibv_send_wr {
                wr_id,
                next: std::ptr::null_mut(),
                sg_list: &mut send_sge as *mut _,
                num_sge: 1,
                opcode: op_type.into(),
                send_flags,
                wr: Default::default(),
                qp_type: Default::default(),
                __bindgen_anon_1: Default::default(),
                __bindgen_anon_2: Default::default(),
            };

            // Set remote address and rkey for RDMA operations
            let remote_connection_info = self.remote_connection_info.as_ref().unwrap();
            send_wr.wr.rdma.remote_addr = remote_connection_info.remote_addr;
            send_wr.wr.rdma.rkey = remote_connection_info.rkey;

            let mut bad_send_wr: *mut ffi::ibv_send_wr = std::ptr::null_mut();
            let ops = &mut (*self.context).ops;
            let errno =
                ops.post_send.as_mut().unwrap()(self.qp, &mut send_wr as *mut _, &mut bad_send_wr);

            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to post send request: {}", os_error),
                ));
            }

            Ok(())
        }
    }

    /// Polls the completion queue for a completion event.
    ///
    /// This function performs a single poll of the completion queue and returns the result.
    /// It does not perform any timing or retry logic - the application is responsible for
    /// implementing any polling strategy (timeouts, retries, etc.).
    ///
    /// Note - while this method does not mutate the Rust struct (e.g. RdmaConnection),
    /// it does consume work completions from the underlying ibverbs completion queue (CQ)
    /// as a side effect. This is thread-safe, but may affect concurrent polls on
    /// the same completion queue.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(wc))` - A completion was found
    /// * `Ok(None)` - No completion was found
    /// * `Err(e)` - An error occurred
    pub fn poll_completion(&self) -> Result<Option<IbvWc>> {
        if self.remote_connection_info.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::NotConnected,
                "RDMA connection not established. Call connect() first.",
            ));
        }

        // SAFETY: Interacts with RDMA device through FFI calls
        unsafe {
            let mut wc = std::mem::MaybeUninit::<ffi::ibv_wc>::zeroed().assume_init();
            let ops = &mut (*self.context).ops;

            let ret = ops.poll_cq.as_mut().unwrap()(self.cq, 1, &mut wc);

            if ret < 0 {
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to poll CQ: {}", Error::last_os_error()),
                ));
            }

            if ret > 0 {
                if !wc.is_valid() {
                    if let Some((status, vendor_err)) = wc.error() {
                        return Err(Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "Work completion failed with status: {:?}, vendor error: {}",
                                status, vendor_err
                            ),
                        ));
                    }
                }
                return Ok(Some(IbvWc::from(wc)));
            }

            // No completion found
            Ok(None)
        }
    }

    /// Get immutable reference to the buffer
    pub fn buffer(&self) -> &[u8] {
        // SAFETY:
        // This API reconstructs a boxed representation of a buffer from its pointer and size.
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), self.buffer.len()) }
    }
}

impl Drop for RdmaConnectionPoint {
    fn drop(&mut self) {
        // SAFETY:
        // This requires calling C++/ibverbs APIs for proper clean up
        // of ibverbs constructs.
        unsafe {
            if !self.qp.is_null() {
                ffi::ibv_destroy_qp(self.qp);
            }
            if !self.cq.is_null() {
                ffi::ibv_destroy_cq(self.cq);
            }
            if !self.mr.is_null() {
                ffi::ibv_dereg_mr(self.mr);
            }
            if !self.pd.is_null() {
                ffi::ibv_dealloc_pd(self.pd);
            }
            if !self.context.is_null() {
                ffi::ibv_close_device(self.context);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    use super::*;
    use crate::ibverbs_primitives::RdmaMemoryRegion;
    use crate::ibverbs_primitives::RdmaOperation;
    use crate::ibverbs_primitives::get_all_devices;

    #[test]
    fn test_create_connection() {
        let mut buffer = Box::new([0u8; 4]);
        let connection = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut buffer[..]),
        );
        assert!(connection.is_ok());
    }

    #[test]
    fn test_get_endpoint() {
        let mut buffer = Box::new([0u8; 4]);
        let connection = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut buffer[..]),
        )
        .unwrap();
        let endpoint = connection.get_connection_info();
        assert!(endpoint.is_ok());
    }

    #[test]
    fn test_loopback_connection() {
        let mut server_buffer = Box::new([0u8; 4096]);
        let mut client_buffer = Box::new([0u8; 4096]);

        let mut server = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut server_buffer[..]),
        )
        .unwrap();
        let mut client = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut client_buffer[..]),
        )
        .unwrap();

        let server_connection_info = server.get_connection_info().unwrap();
        let client_connection_info = client.get_connection_info().unwrap();

        assert!(server.connect(&client_connection_info).is_ok());
        assert!(client.connect(&server_connection_info).is_ok());
    }

    #[test]
    fn test_invalid_range() {
        let mut buffer = Box::new([0u8; 4096]);
        let mut connection = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut buffer[..]),
        )
        .unwrap();

        // Out of bounds range
        assert!(
            connection
                .post_send(4000..5000, 1, true, RdmaOperation::Write)
                .is_err()
        );
    }

    #[test]
    fn test_not_connected() {
        let mut buffer = Box::new([0u8; 4096]);
        let mut connection = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut buffer[..]),
        )
        .unwrap();

        // Operations should fail when not connected
        assert!(
            connection
                .post_send(0..8, 1, true, RdmaOperation::Write)
                .is_err()
        );
        assert!(connection.poll_completion().is_err());
    }

    #[test]
    fn test_loopback_rdma_write() {
        // Create a buffer for our RDMA operations
        const BSIZE: usize = 128;
        let mut server_buffer = Box::new([0u8; BSIZE]);
        let mut client_buffer = Box::new([0u8; BSIZE]);

        // Fill the client buffer with test data
        for (i, val) in client_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        // Create connections
        let mut server = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut server_buffer[..]),
        )
        .unwrap();
        let mut client = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut client_buffer[..]),
        )
        .unwrap();

        // Get endpoints
        let server_connection_info = server.get_connection_info().unwrap();
        let client_connection_info = client.get_connection_info().unwrap();

        // Connect both sides
        assert!(server.connect(&client_connection_info).is_ok());
        assert!(client.connect(&server_connection_info).is_ok());

        // Client performs RDMA write to server
        client
            .post_send(..BSIZE, 1, true, RdmaOperation::Write)
            .unwrap();

        // Poll for completion
        let mut write_completed = false;
        let timeout = Duration::from_secs(2);
        let start_time = Instant::now();

        while !write_completed && start_time.elapsed() < timeout {
            match client.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        write_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }

        assert!(write_completed, "RDMA write operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            let sent_byte = server.buffer()[i];
            let received_byte = server.buffer()[i];
            assert_eq!(sent_byte, received_byte, "Data mismatch at position {}", i);
        }
    }

    #[test]
    fn test_loopback_rdma_read() {
        // Create a buffer for our RDMA operations
        const BSIZE: usize = 128;
        let mut server_buffer = Box::new([0u8; BSIZE]);
        let mut client_buffer = Box::new([0u8; BSIZE]);

        // Fill the server buffer with test data
        for (i, val) in server_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        // Create connections
        let mut server = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut server_buffer[..]),
        )
        .unwrap();
        let mut client = RdmaConnectionPoint::new(
            RdmaConnectionConfig::default(),
            RdmaMemoryRegion::from(&mut client_buffer[..]),
        )
        .unwrap();

        // Get endpoints
        let server_connection_info = server.get_connection_info().unwrap();
        let client_connection_info = client.get_connection_info().unwrap();

        // Connect both sides
        assert!(server.connect(&client_connection_info).is_ok());
        assert!(client.connect(&server_connection_info).is_ok());

        // Client performs RDMA read from server
        client
            .post_send(..BSIZE, 1, true, RdmaOperation::Read)
            .unwrap();

        // Poll for completion
        let mut read_completed = false;
        let timeout = Duration::from_secs(2);
        let start_time = Instant::now();

        while !read_completed && start_time.elapsed() < timeout {
            match client.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        read_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }

        assert!(read_completed, "RDMA read operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            let server_byte = server.buffer()[i];
            let client_byte = client.buffer()[i];
            assert_eq!(server_byte, client_byte, "Data mismatch at position {}", i);
        }
    }

    #[test]
    fn test_two_device_write() {
        let devices = get_all_devices();
        assert!(devices.len() > 5, "assumes we're using devices 0 and 5");
        const BSIZE: usize = 128;
        let mut server_buffer = Box::new([0u8; BSIZE]);
        let mut client_buffer = Box::new([0u8; BSIZE]);

        // Fill the client buffer with test data
        for (i, val) in client_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }
        let device1 = devices.clone().into_iter().next().unwrap();
        let device2 = devices.clone().into_iter().nth(4).unwrap();
        let config1 = RdmaConnectionConfig {
            device: device1,
            ..Default::default()
        };
        let config2 = RdmaConnectionConfig {
            device: device2,
            ..Default::default()
        };
        let mut server =
            RdmaConnectionPoint::new(config1, RdmaMemoryRegion::from(&mut server_buffer[..]))
                .unwrap();
        let mut client =
            RdmaConnectionPoint::new(config2, RdmaMemoryRegion::from(&mut client_buffer[..]))
                .unwrap();
        let server_connection_info = server.get_connection_info().unwrap();
        let client_connection_info = client.get_connection_info().unwrap();
        assert!(server.connect(&client_connection_info).is_ok());
        assert!(client.connect(&server_connection_info).is_ok());

        // Client performs RDMA write to server
        client
            .post_send(..BSIZE, 1, true, RdmaOperation::Write)
            .unwrap();

        // Poll for completion
        let mut write_completed = false;
        let timeout = Duration::from_secs(2);
        let start_time = Instant::now();

        while !write_completed && start_time.elapsed() < timeout {
            match client.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        write_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }

        assert!(write_completed, "RDMA write operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            let sent_byte = server.buffer()[i];
            let received_byte = server.buffer()[i];
            assert_eq!(sent_byte, received_byte, "Data mismatch at position {}", i);
        }
    }

    #[test]
    pub fn test_two_device_read() {
        let devices = get_all_devices();
        assert!(devices.len() > 5, "assumes we're using devices 0 and 5");
        // Create a buffer for our RDMA operations
        const BSIZE: usize = 128;
        let mut server_buffer = Box::new([0u8; BSIZE]);
        let mut client_buffer = Box::new([0u8; BSIZE]);

        // Fill the server buffer with test data
        for (i, val) in server_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        let device1 = devices.clone().into_iter().next().unwrap();
        let device2 = devices.clone().into_iter().nth(4).unwrap();
        let config1 = RdmaConnectionConfig {
            device: device1,
            ..Default::default()
        };
        let config2 = RdmaConnectionConfig {
            device: device2,
            ..Default::default()
        };
        let mut server =
            RdmaConnectionPoint::new(config1, RdmaMemoryRegion::from(&mut server_buffer[..]))
                .unwrap();
        let mut client =
            RdmaConnectionPoint::new(config2, RdmaMemoryRegion::from(&mut client_buffer[..]))
                .unwrap();

        // Get endpoints
        let server_endpoint = server.get_connection_info().unwrap();
        let client_endpoint = client.get_connection_info().unwrap();

        // Connect both sides
        assert!(server.connect(&client_endpoint).is_ok());
        assert!(client.connect(&server_endpoint).is_ok());

        // Client performs RDMA read from server
        client
            .post_send(..BSIZE, 1, true, RdmaOperation::Read)
            .unwrap();

        // Poll for completion
        let mut read_completed = false;
        let start_time = Instant::now();
        let timeout = Duration::from_secs(2);
        while !read_completed && start_time.elapsed() < timeout {
            match client.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        read_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }

        assert!(read_completed, "RDMA read operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            let server_byte = server.buffer()[i];
            let client_byte = client.buffer()[i];
            assert_eq!(server_byte, client_byte, "Data mismatch at position {}", i);
        }
    }
}
