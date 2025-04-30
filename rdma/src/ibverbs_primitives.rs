//! This file contains primitive data structures for interacting with ibverbs.
//!
//! Primitives:
//! - `RdmaDevice`: Represents an RDMA device, i.e. 'mlx5_0'. Contains information about the device, such as:
//!   its name, vendor ID, vendor part ID, hardware version, firmware version, node GUID, and capabilities.
//! - `RdmaPort`: Represents information about the port of an RDMA device, including state, physical state,
//!   LID (Local Identifier), and GID (Global Identifier) information.
//! - `RdmaMemoryRegion`: Represents a memory region that can be registered with an RDMA device for direct
//!   memory access operations.
//! - `RdmaOperation`: Represents the type of RDMA operation to perform (Read or Write).
//! - `RdmaConnectionInfo`: Contains connection information needed to establish an RDMA connection with a remote endpoint.
//! - `IbvWc`: Wrapper around ibverbs work completion structure, used to track the status of RDMA operations.
use std::ffi::CStr;
use std::fmt;

use hyperactor::Named;
use ibverbs::Gid;
use serde::ser::SerializeStruct;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RdmaDevice {
    name: String,
    vendor_id: u32,
    vendor_part_id: u32,
    hw_ver: u32,
    fw_ver: String,
    node_guid: u64,
    ports: Vec<RdmaPort>,
    max_qp: i32,
    max_cq: i32,
    max_mr: i32,
    max_pd: i32,
    max_qp_wr: i32,
    max_sge: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RdmaPort {
    port_num: u8,
    state: String,
    physical_state: String,
    base_lid: u16,
    lmc: u8,
    sm_lid: u16,
    capability_mask: u32,
    link_layer: String,
    gid: String,
    gid_tbl_len: i32,
}

impl fmt::Display for RdmaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.name)?;
        writeln!(f, "\tNumber of ports: {}", self.ports.len())?;
        writeln!(f, "\tFirmware version: {}", self.fw_ver)?;
        writeln!(f, "\tHardware version: {}", self.hw_ver)?;
        writeln!(f, "\tNode GUID: 0x{:016x}", self.node_guid)?;
        writeln!(f, "\tVendor ID: 0x{:x}", self.vendor_id)?;
        writeln!(f, "\tVendor part ID: {}", self.vendor_part_id)?;
        writeln!(f, "\tMax QPs: {}", self.max_qp)?;
        writeln!(f, "\tMax CQs: {}", self.max_cq)?;
        writeln!(f, "\tMax MRs: {}", self.max_mr)?;
        writeln!(f, "\tMax PDs: {}", self.max_pd)?;
        writeln!(f, "\tMax QP WRs: {}", self.max_qp_wr)?;
        writeln!(f, "\tMax SGE: {}", self.max_sge)?;

        for port in &self.ports {
            write!(f, "{}", port)?;
        }

        Ok(())
    }
}

impl fmt::Display for RdmaPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\tPort {}:", self.port_num)?;
        writeln!(f, "\t\tState: {}", self.state)?;
        writeln!(f, "\t\tPhysical state: {}", self.physical_state)?;
        writeln!(f, "\t\tBase lid: {}", self.base_lid)?;
        writeln!(f, "\t\tLMC: {}", self.lmc)?;
        writeln!(f, "\t\tSM lid: {}", self.sm_lid)?;
        writeln!(f, "\t\tCapability mask: 0x{:08x}", self.capability_mask)?;
        writeln!(f, "\t\tLink layer: {}", self.link_layer)?;
        writeln!(f, "\t\tGID: {}", self.gid)?;
        writeln!(f, "\t\tGID table length: {}", self.gid_tbl_len)?;
        Ok(())
    }
}

pub fn get_port_state_str(state: ffi::ibv_port_state::Type) -> String {
    // SAFETY: We are calling a C function that returns a C string.
    unsafe {
        let c_str = ffi::ibv_port_state_str(state);
        if c_str.is_null() {
            return "Unknown".to_string();
        }
        CStr::from_ptr(c_str).to_string_lossy().into_owned()
    }
}

pub fn get_port_phy_state_str(phys_state: u8) -> String {
    match phys_state {
        1 => "Sleep".to_string(),
        2 => "Polling".to_string(),
        3 => "Disabled".to_string(),
        4 => "PortConfigurationTraining".to_string(),
        5 => "LinkUp".to_string(),
        6 => "LinkErrorRecovery".to_string(),
        7 => "PhyTest".to_string(),
        _ => "No state change".to_string(),
    }
}

pub fn get_link_layer_str(link_layer: u8) -> String {
    match link_layer {
        1 => "InfiniBand".to_string(),
        2 => "Ethernet".to_string(),
        _ => "Unknown".to_string(),
    }
}

pub fn format_gid(gid: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}",
        gid[0],
        gid[1],
        gid[2],
        gid[3],
        gid[4],
        gid[5],
        gid[6],
        gid[7],
        gid[8],
        gid[9],
        gid[10],
        gid[11],
        gid[12],
        gid[13],
        gid[14],
        gid[15]
    )
}

/// Retrieves information about all available RDMA devices in the system.
///
/// This function queries the system for all available RDMA devices and returns
/// detailed information about each device, including its capabilities, ports,
/// and attributes.
///
/// # Returns
///
/// A vector of `RdmaDevice` structures, each representing an RDMA device in the system.
/// Returns an empty vector if no devices are found or if there was an error querying
/// the devices.
pub fn get_all_devices() -> Vec<RdmaDevice> {
    let mut devices = Vec::new();

    // SAFETY: We are calling several C functions from libibverbs.
    unsafe {
        let mut num_devices = 0;
        let device_list = ffi::ibv_get_device_list(&mut num_devices);
        if device_list.is_null() || num_devices == 0 {
            return devices;
        }

        for i in 0..num_devices {
            let device = *device_list.add(i as usize);
            if device.is_null() {
                continue;
            }

            let context = ffi::ibv_open_device(device);
            if context.is_null() {
                continue;
            }

            let device_name = CStr::from_ptr(ffi::ibv_get_device_name(device))
                .to_string_lossy()
                .into_owned();

            let mut device_attr = ffi::ibv_device_attr::default();
            if ffi::ibv_query_device(context, &mut device_attr) != 0 {
                ffi::ibv_close_device(context);
                continue;
            }

            let fw_ver = CStr::from_ptr(device_attr.fw_ver.as_ptr())
                .to_string_lossy()
                .into_owned();

            let mut rdma_device = RdmaDevice {
                name: device_name,
                vendor_id: device_attr.vendor_id,
                vendor_part_id: device_attr.vendor_part_id,
                hw_ver: device_attr.hw_ver,
                fw_ver,
                node_guid: device_attr.node_guid,
                ports: Vec::new(),
                max_qp: device_attr.max_qp,
                max_cq: device_attr.max_cq,
                max_mr: device_attr.max_mr,
                max_pd: device_attr.max_pd,
                max_qp_wr: device_attr.max_qp_wr,
                max_sge: device_attr.max_sge,
            };

            for port_num in 1..=device_attr.phys_port_cnt {
                let mut port_attr = ffi::ibv_port_attr::default();
                if ffi::ibv_query_port(
                    context,
                    port_num,
                    &mut port_attr as *mut ffi::ibv_port_attr as *mut _,
                ) != 0
                {
                    continue;
                }
                let state = get_port_state_str(port_attr.state);
                let physical_state = get_port_phy_state_str(port_attr.phys_state);

                let link_layer = get_link_layer_str(port_attr.link_layer);

                let mut gid = ffi::ibv_gid::default();
                let gid_str = if ffi::ibv_query_gid(context, port_num, 0, &mut gid) == 0 {
                    format_gid(&gid.raw)
                } else {
                    "N/A".to_string()
                };

                let rdma_port = RdmaPort {
                    port_num,
                    state,
                    physical_state,
                    base_lid: port_attr.lid,
                    lmc: port_attr.lmc,
                    sm_lid: port_attr.sm_lid,
                    capability_mask: port_attr.port_cap_flags,
                    link_layer,
                    gid: gid_str,
                    gid_tbl_len: port_attr.gid_tbl_len,
                };

                rdma_device.ports.push(rdma_port);
            }

            devices.push(rdma_device);
            ffi::ibv_close_device(context);
        }

        ffi::ibv_free_device_list(device_list);
    }

    devices
}

/// Represents a memory region that can be registered with an RDMA device.
///
/// An `RdmaMemoryRegion` encapsulates a pointer to a memory buffer and its size.
/// This memory region can be registered with an RDMA device to allow direct memory
/// access operations (such as RDMA reads and writes) to be performed on it.
///
/// # Safety
///
/// The memory pointed to by `ptr` must remain valid for the lifetime of the `RdmaMemoryRegion`.
/// The caller is responsible for ensuring that the memory is not freed or moved while
/// RDMA operations are in progress.
#[derive(Debug, Clone)]
pub struct RdmaMemoryRegion {
    ptr: *mut u8,
    size: usize,
}

// SAFETY: We guarantee that the pointer is only accessed
// in thread-safe ways and that proper synchronization is handled
// properly through libibverbs.
unsafe impl Send for RdmaMemoryRegion {}

// SAFETY: We ensure that the memory region is only accessed in thread-safe ways.
// Multiple threads can safely read from the memory region simultaneously,
// and any mutations are properly synchronized through the RDMA operations
// which are handled by the hardware and libibverbs library.
unsafe impl Sync for RdmaMemoryRegion {}

impl<'a> From<&'a mut [u8]> for RdmaMemoryRegion {
    fn from(buffer: &'a mut [u8]) -> Self {
        let ptr = buffer.as_mut_ptr();
        let size = buffer.len();
        RdmaMemoryRegion { ptr, size }
    }
}

impl RdmaMemoryRegion {
    /// Creates a new `RdmaMemoryRegion` with the given pointer and size.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to the memory buffer
    /// * `size` - Size of the memory buffer in bytes
    ///
    /// # Safety
    ///
    /// The caller must ensure that the memory pointed to by `ptr` remains valid
    /// for the lifetime of the `RdmaMemoryRegion`.
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        RdmaMemoryRegion { ptr, size }
    }

    /// Returns a pointer to the memory buffer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the size of the memory buffer in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the memory buffer is empty (size is 0).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// SAFETY CONCERN: Serializing/deserializing raw pointers is unsafe and non-portable.
// This implementation is a temporary workaround to make RdmaMemoryRegion compatible
// with hyperactor. The proper solution would be to refactor the code
// to use handles or identifiers instead of raw pointers when communicating between
// processes or across network boundaries.
impl serde::Serialize for RdmaMemoryRegion {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("RdmaMemoryRegion", 2)?;
        state.serialize_field("ptr", &(self.ptr as u64))?;
        state.serialize_field("size", &self.size)?;
        state.end()
    }
}

struct RdmaMemoryRegionVisitor;

impl<'de> serde::de::Visitor<'de> for RdmaMemoryRegionVisitor {
    type Value = RdmaMemoryRegion;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a struct representing a pointer and buffer length")
    }

    fn visit_seq<V>(self, mut seq: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: serde::de::SeqAccess<'de>,
    {
        let ptr = seq
            .next_element::<u64>()?
            .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        let len = seq
            .next_element::<usize>()?
            .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
        Ok(RdmaMemoryRegion {
            ptr: ptr as *mut u8,
            size: len,
        })
    }
}

impl<'de> serde::Deserialize<'de> for RdmaMemoryRegion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "RdmaMemoryRegion",
            &["ptr", "len"],
            RdmaMemoryRegionVisitor,
        )
    }
}

/// Enum representing the common RDMA operations.
///
/// This provides a more ergonomic interface to the underlying ibv_wr_opcode types.
/// RDMA operations allow for direct memory access between two machines without
/// involving the CPU of the target machine.
///
/// # Variants
///
/// * `Write` - Represents an RDMA write operation where data is written from the local
///   memory to a remote memory region.
/// * `Read` - Represents an RDMA read operation where data is read from a remote memory
///   region into the local memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdmaOperation {
    /// RDMA write operation
    Write,
    /// RDMA read operation
    Read,
}

impl From<RdmaOperation> for ffi::ibv_wr_opcode::Type {
    fn from(op: RdmaOperation) -> Self {
        match op {
            RdmaOperation::Write => ffi::ibv_wr_opcode::IBV_WR_RDMA_WRITE,
            RdmaOperation::Read => ffi::ibv_wr_opcode::IBV_WR_RDMA_READ,
        }
    }
}

impl From<ffi::ibv_wc_opcode::Type> for RdmaOperation {
    fn from(op: ffi::ibv_wc_opcode::Type) -> Self {
        match op {
            ffi::ibv_wc_opcode::IBV_WC_RDMA_WRITE => RdmaOperation::Write,
            ffi::ibv_wc_opcode::IBV_WC_RDMA_READ => RdmaOperation::Read,
            _ => panic!("Unsupported operation type"),
        }
    }
}

/// Contains connection information needed to establish an RDMA connection with a remote endpoint.
///
/// An `RdmaConnectionInfo` encapsulates all the necessary information to establish a connection
/// with a remote RDMA device. This includes queue pair number, LID (Local Identifier),
/// GID (Global Identifier), remote memory address, remote key, and packet sequence number.
///
/// # Fields
///
/// * `qp_num` - Queue Pair Number, uniquely identifies a queue pair on the remote device
/// * `lid` - Local Identifier, used for addressing in InfiniBand subnet
/// * `gid` - Global Identifier, used for routing across subnets (similar to IPv6 address)
/// * `remote_addr` - Address of the remote memory region
/// * `rkey` - Remote key, used to access the remote memory region
/// * `psn` - Packet Sequence Number, used for ordering packets
#[derive(Default, Named, Clone, serde::Serialize, serde::Deserialize)]
pub struct RdmaConnectionInfo {
    pub qp_num: u32,
    pub lid: u16,
    pub gid: Option<Gid>,
    pub remote_addr: u64,
    pub rkey: u32,
    pub psn: u32,
}

impl std::fmt::Debug for RdmaConnectionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RdmaConnectionInfo {{ qp_num: {}, lid: {}, gid: {:?}, psn: 0x{:x} }}",
            self.qp_num, self.lid, self.gid, self.psn
        )
    }
}

/// Wrapper around ibv_wc (ibverbs work completion).
///
/// This exposes only the public fields of ffi::ibv_wc, allowing us to more easily
/// interact with it from Rust. Work completions are used to track the status of
/// RDMA operations and are generated when an operation completes.
///
/// # Fields
///
/// * `wr_id` - Work Request ID, used to identify the completed operation
/// * `len` - Length of the data transferred
/// * `valid` - Whether the work completion is valid
/// * `error` - Error information if the operation failed
/// * `opcode` - Type of operation that completed (read, write, etc.)
/// * `bytes` - Immediate data (if any)
/// * `qp_num` - Queue Pair Number
/// * `src_qp` - Source Queue Pair Number
/// * `pkey_index` - Partition Key Index
/// * `slid` - Source LID
/// * `sl` - Service Level
/// * `dlid_path_bits` - Destination LID Path Bits
#[derive(Debug, Named, Clone, serde::Serialize, serde::Deserialize)]
pub struct IbvWc {
    wr_id: u64,
    len: usize,
    valid: bool,
    error: Option<(ffi::ibv_wc_status::Type, u32)>,
    opcode: ffi::ibv_wc_opcode::Type,
    bytes: Option<u32>,
    qp_num: u32,
    src_qp: u32,
    pkey_index: u16,
    slid: u16,
    sl: u8,
    dlid_path_bits: u8,
}

impl From<ffi::ibv_wc> for IbvWc {
    fn from(wc: ffi::ibv_wc) -> Self {
        IbvWc {
            wr_id: wc.wr_id(),
            len: wc.len(),
            valid: wc.is_valid(),
            error: wc.error(),
            opcode: wc.opcode(),
            bytes: wc.imm_data(),
            qp_num: wc.qp_num,
            src_qp: wc.src_qp,
            pkey_index: wc.pkey_index,
            slid: wc.slid,
            sl: wc.sl,
            dlid_path_bits: wc.dlid_path_bits,
        }
    }
}

impl IbvWc {
    /// Returns the Work Request ID associated with this work completion.
    ///
    /// The Work Request ID is used to identify the specific operation that completed.
    /// It is set by the application when posting the work request and is returned
    /// unchanged in the work completion.
    pub fn wr_id(&self) -> u64 {
        self.wr_id
    }

    /// Returns whether this work completion is valid.
    ///
    /// A valid work completion indicates that the operation completed successfully.
    /// If false, the `error` field may contain additional information about the failure.
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

impl RdmaDevice {
    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn first_available() -> Option<RdmaDevice> {
        let devices = get_all_devices();
        if devices.is_empty() {
            None
        } else {
            Some(devices.into_iter().next().unwrap())
        }
    }

    pub fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    pub fn vendor_part_id(&self) -> u32 {
        self.vendor_part_id
    }

    pub fn hw_ver(&self) -> u32 {
        self.hw_ver
    }

    pub fn fw_ver(&self) -> &String {
        &self.fw_ver
    }

    pub fn node_guid(&self) -> u64 {
        self.node_guid
    }

    pub fn ports(&self) -> &Vec<RdmaPort> {
        &self.ports
    }

    pub fn max_qp(&self) -> i32 {
        self.max_qp
    }

    pub fn max_cq(&self) -> i32 {
        self.max_cq
    }

    pub fn max_mr(&self) -> i32 {
        self.max_mr
    }

    pub fn max_pd(&self) -> i32 {
        self.max_pd
    }

    pub fn max_qp_wr(&self) -> i32 {
        self.max_qp_wr
    }

    pub fn max_sge(&self) -> i32 {
        self.max_sge
    }
}

impl Default for RdmaDevice {
    fn default() -> Self {
        get_all_devices()
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("No RDMA devices found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_all_devices() {
        let devices = get_all_devices();
        assert!(!devices.is_empty(), "No RDMA devices found");

        // Basic validation of first device
        let device = &devices[0];
        assert!(!device.name().is_empty(), "Device name should not be empty");
        assert!(
            !device.ports().is_empty(),
            "Device should have at least one port"
        );
    }

    #[test]
    fn test_first_available() {
        let device = RdmaDevice::first_available();
        assert!(device.is_some(), "Should find at least one RDMA device");

        if let Some(dev) = device {
            // Verify getters return expected values
            assert_eq!(dev.vendor_id(), dev.vendor_id);
            assert_eq!(dev.vendor_part_id(), dev.vendor_part_id);
            assert_eq!(dev.hw_ver(), dev.hw_ver);
            assert_eq!(dev.fw_ver(), &dev.fw_ver);
            assert_eq!(dev.node_guid(), dev.node_guid);
            assert_eq!(dev.max_qp(), dev.max_qp);
            assert_eq!(dev.max_cq(), dev.max_cq);
            assert_eq!(dev.max_mr(), dev.max_mr);
            assert_eq!(dev.max_pd(), dev.max_pd);
            assert_eq!(dev.max_qp_wr(), dev.max_qp_wr);
            assert_eq!(dev.max_sge(), dev.max_sge);
        }
    }

    #[test]
    fn test_device_display() {
        if let Some(device) = RdmaDevice::first_available() {
            let display_output = format!("{}", device);
            assert!(
                display_output.contains(&device.name),
                "Display should include device name"
            );
            assert!(
                display_output.contains(&device.fw_ver),
                "Display should include firmware version"
            );
        }
    }

    #[test]
    fn test_port_display() {
        if let Some(device) = RdmaDevice::first_available() {
            if !device.ports().is_empty() {
                let port = &device.ports()[0];
                let display_output = format!("{}", port);
                assert!(
                    display_output.contains(&port.state),
                    "Display should include port state"
                );
                assert!(
                    display_output.contains(&port.link_layer),
                    "Display should include link layer"
                );
            }
        }
    }

    #[test]
    fn test_rdma_memory_region() {
        let mut buffer = vec![0u8; 1024];
        let mr = RdmaMemoryRegion::from(&mut buffer[..]);

        assert_eq!(mr.len(), 1024);
        assert_eq!(mr.as_ptr(), buffer.as_mut_ptr());
        assert!(!mr.is_empty());

        let empty_mr = RdmaMemoryRegion::new(std::ptr::null_mut(), 0);
        assert_eq!(empty_mr.len(), 0);
        assert!(empty_mr.is_empty());
    }

    #[test]
    fn test_rdma_operation_conversion() {
        assert_eq!(
            ffi::ibv_wr_opcode::IBV_WR_RDMA_WRITE,
            ffi::ibv_wr_opcode::Type::from(RdmaOperation::Write)
        );
        assert_eq!(
            ffi::ibv_wr_opcode::IBV_WR_RDMA_READ,
            ffi::ibv_wr_opcode::Type::from(RdmaOperation::Read)
        );

        assert_eq!(
            RdmaOperation::Write,
            RdmaOperation::from(ffi::ibv_wc_opcode::IBV_WC_RDMA_WRITE)
        );
        assert_eq!(
            RdmaOperation::Read,
            RdmaOperation::from(ffi::ibv_wc_opcode::IBV_WC_RDMA_READ)
        );
    }

    #[test]
    fn test_rdma_endpoint() {
        let endpoint = RdmaConnectionInfo {
            qp_num: 42,
            lid: 123,
            gid: None,
            remote_addr: 0x1000,
            rkey: 0xabcd,
            psn: 0x5678,
        };

        let debug_str = format!("{:?}", endpoint);
        assert!(debug_str.contains("qp_num: 42"));
        assert!(debug_str.contains("lid: 123"));
        assert!(debug_str.contains("psn: 0x5678"));
    }

    #[test]
    fn test_ibv_wc() {
        let mut wc = ffi::ibv_wc::default();

        // SAFETY: modifies private fields through pointer manipulation
        unsafe {
            // Cast to pointer and modify the fields directly
            let wc_ptr = &mut wc as *mut ffi::ibv_wc as *mut u8;

            // Set wr_id (at offset 0, u64)
            *(wc_ptr as *mut u64) = 42;

            // Set status to SUCCESS (at offset 8, u32)
            *(wc_ptr.add(8) as *mut i32) = ffi::ibv_wc_status::IBV_WC_SUCCESS as i32;
        }
        let ibv_wc = IbvWc::from(wc);
        assert_eq!(ibv_wc.wr_id(), 42);
        assert!(ibv_wc.is_valid());
    }

    #[test]
    fn test_format_gid() {
        let gid = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];

        let formatted = format_gid(&gid);
        assert_eq!(formatted, "1234:5678:9abc:def0:1122:3344:5566:7788");
    }
}
