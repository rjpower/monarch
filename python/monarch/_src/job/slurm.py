# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import subprocess
import sys
from typing import cast, Dict, List, Optional, Sequence

from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure

from monarch._src.actor.bootstrap import attach_to_workers
from monarch._src.actor.host_mesh import HostMesh
from monarch._src.job.job import JobState, JobTrait


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.propagate = False


class SlurmJob(JobTrait):
    """
    A job scheduler that uses SLURM command line tools to schedule jobs.

    This implementation:
    1. Uses sbatch to submit SLURM jobs that start monarch workers
    2. Queries job status with squeue to get allocated hostnames
    3. Uses the hostnames to connect to the started workers

    Unlike LoginJob, this submits batch jobs that can allocate multiple nodes.
    """

    def __init__(
        self,
        meshes: Dict[str, int],  # mesh_name -> number of nodes
        python_exe: str = "python",
        slurm_args: Sequence[str] = (),
        monarch_port: int = 22222,
        job_name: str = "monarch_job",
        ntasks_per_node: int = 1,
        time_limit: str = "01:00:00",
        partition: Optional[str] = None,
    ) -> None:
        configure(default_transport=ChannelTransport.Tcp)
        self._meshes = meshes
        self._python_exe = python_exe
        self._slurm_args = slurm_args
        self._port = monarch_port
        self._job_name = job_name
        self._ntasks_per_node = ntasks_per_node
        self._time_limit = time_limit
        self._partition = partition
        # Track the single SLURM job ID and all allocated hostnames
        self._slurm_job_id: Optional[str] = None
        self._all_hostnames: List[str] = []
        super().__init__()

    def add_mesh(self, name: str, num_nodes: int) -> None:
        """Add a host mesh with the specified number of nodes."""
        self._meshes[name] = num_nodes

    def _create(self, client_script: Optional[str]) -> None:
        """Submit a single SLURM job for all meshes."""
        if client_script is not None:
            raise RuntimeError("SlurmJob cannot run batch-mode scripts")

        # Calculate total nodes needed across all meshes
        total_nodes = sum(self._meshes.values())

        # Submit a single SLURM job for all nodes
        self._slurm_job_id = self._submit_slurm_job(total_nodes)

    def _submit_slurm_job(self, num_nodes: int) -> str:
        """Submit a SLURM job for all nodes."""
        # Create a unique job name
        unique_job_name = f"{self._job_name}_{os.getpid()}"

        # Build the sbatch command
        sbatch_cmd = [
            "sbatch",
            "--job-name",
            unique_job_name,
            "--ntasks-per-node",
            str(self._ntasks_per_node),
            "--time",
            self._time_limit,
            "--nodes",
            str(num_nodes),
            "--output",
            f"/tmp/slurm_%j_{unique_job_name}.out",
            "--error",
            f"/tmp/slurm_%j_{unique_job_name}.err",
        ]

        # Add partition if specified
        if self._partition:
            sbatch_cmd.extend(["--partition", self._partition])

        # Add any additional SLURM arguments
        sbatch_cmd.extend(self._slurm_args)

        # Create the Python command to run on each allocated node
        python_command = f'import socket; from monarch.actor import run_worker_loop_forever; hostname = socket.gethostname(); run_worker_loop_forever(address=f"tcp://{{hostname}}:{self._port}", ca="trust_all_connections")'

        # Submit the job
        logger.info(f"Submitting SLURM job with {num_nodes} nodes")

        # Add the Python command as the job to execute
        sbatch_cmd.extend([self._python_exe, "-c", python_command])

        try:
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the job ID from sbatch output (typically "Submitted batch job 12345")
            job_id = None
            for line in result.stdout.strip().split("\n"):
                if "Submitted batch job" in line:
                    job_id = line.split()[-1]
                    break

            if not job_id:
                raise RuntimeError(
                    f"Failed to parse job ID from sbatch output: {result.stdout}"
                )

            logger.info(f"SLURM job {job_id} submitted")
            return job_id

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to submit SLURM job: {e.stderr}") from e

    def _wait_for_job_start(
        self, job_id: str, expected_nodes: int, timeout: int = 300
    ) -> List[str]:
        """
        Wait for the SLURM job to start and return the allocated hostnames.

        Args:
            job_id: The SLURM job ID
            expected_nodes: Expected number of nodes to be allocated
            timeout: Maximum time to wait in seconds

        Returns:
            List of hostnames of the allocated nodes
        """
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Use squeue to check job status and get hostname
                result = subprocess.run(
                    ["squeue", "--job", job_id, "--format", "%T,%N", "--noheader"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.stdout.strip():
                    status, nodelist = result.stdout.strip().split(",", 1)

                    if status == "RUNNING":
                        # Parse the nodelist to get all hostnames
                        hostnames = self._parse_nodelist(nodelist)
                        logger.info(
                            f"SLURM job {job_id} is running on {len(hostnames)} nodes: {hostnames}"
                        )

                        if len(hostnames) != expected_nodes:
                            logger.warning(
                                f"Expected {expected_nodes} nodes but got {len(hostnames)}"
                            )

                        return hostnames
                    elif status in ["FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED"]:
                        raise RuntimeError(
                            f"SLURM job {job_id} failed with status: {status}"
                        )
                    else:
                        logger.debug(f"SLURM job {job_id} status: {status}, waiting...")

                else:
                    # Job might be completed or not found
                    raise RuntimeError(f"SLURM job {job_id} not found in queue")

            except subprocess.CalledProcessError as e:
                logger.warning(f"Error checking job {job_id} status: {e.stderr}")

            time.sleep(2)  # Check every 2 seconds

        raise RuntimeError(f"Timeout waiting for SLURM job {job_id} to start")

    def _parse_nodelist(self, nodelist: str) -> List[str]:
        """
        Parse SLURM nodelist format and return all hostnames.

        Examples:
        - "node001" -> ["node001"]
        - "node[001-003]" -> ["node001", "node002", "node003"]
        - "gpu01,gpu02" -> ["gpu01", "gpu02"]
        """
        hostnames = []

        # Split by comma first for multiple ranges/hosts
        parts = [part.strip() for part in nodelist.split(",")]

        for part in parts:
            if "[" in part and "]" in part:
                # Handle bracket notation like "node[001-003]" or "node[001,005,010-012]"
                base = part.split("[")[0]
                range_part = part.split("[")[1].split("]")[0]

                # Handle comma-separated list inside brackets
                range_items = [item.strip() for item in range_part.split(",")]

                for item in range_items:
                    if "-" in item:
                        # Handle range like "001-003"
                        start_str, end_str = item.split("-")
                        start_num = int(start_str)
                        end_num = int(end_str)
                        width = len(start_str)  # Preserve leading zeros

                        for num in range(start_num, end_num + 1):
                            hostname = f"{base}{str(num).zfill(width)}"
                            hostnames.append(hostname)
                    else:
                        # Single number in brackets
                        hostname = f"{base}{item}"
                        hostnames.append(hostname)
            else:
                # Simple hostname without brackets
                hostnames.append(part)

        return hostnames

    def _state(self) -> JobState:
        """Get the current state of allocated meshes."""
        if not self._jobs_active():
            raise RuntimeError("SLURM job is no longer active")

        # Wait for job to start and get hostnames if not already done
        if not self._all_hostnames and self._slurm_job_id is not None:
            total_nodes = sum(self._meshes.values())
            self._all_hostnames = self._wait_for_job_start(
                self._slurm_job_id, total_nodes
            )

        # Distribute the allocated hostnames among meshes
        host_meshes = {}
        hostname_idx = 0

        for mesh_name, num_nodes in self._meshes.items():
            # Get the next num_nodes hostnames for this mesh
            mesh_hostnames = self._all_hostnames[
                hostname_idx : hostname_idx + num_nodes
            ]
            hostname_idx += num_nodes

            # Create worker addresses for each hostname
            workers = [f"tcp://{hostname}:{self._port}" for hostname in mesh_hostnames]
            host_mesh = cast(
                "HostMesh",
                attach_to_workers(
                    name=mesh_name,
                    ca="trust_all_connections",
                    workers=workers,  # type: ignore[arg-type]
                ),
            )
            host_meshes[mesh_name] = host_mesh

        return JobState(host_meshes)

    def can_run(self, spec: "JobTrait") -> bool:
        """Check if this job can run the given spec."""
        return (
            isinstance(spec, SlurmJob)
            and spec._meshes == self._meshes
            and spec._python_exe == self._python_exe
            and spec._port == self._port
            and spec._slurm_args == self._slurm_args
            and spec._job_name == self._job_name
            and spec._ntasks_per_node == self._ntasks_per_node
            and spec._time_limit == self._time_limit
            and spec._partition == self._partition
            and self._jobs_active()
        )

    def _jobs_active(self) -> bool:
        """Check if SLURM job is still active by querying squeue."""
        if not self.active or self._slurm_job_id is None:
            return False

        try:
            # Check if the job is still in the queue
            result = subprocess.run(
                ["squeue", "--job", self._slurm_job_id, "--format", "%T", "--noheader"],
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout.strip():
                status = result.stdout.strip()
                if status in [
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "PREEMPTED",
                    "COMPLETED",
                ]:
                    logger.warning(
                        f"SLURM job {self._slurm_job_id} has status: {status}"
                    )
                    return False
            else:
                # Job not in queue anymore
                logger.warning(f"SLURM job {self._slurm_job_id} not found in queue")
                return False

        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Error checking job {self._slurm_job_id} status: {e.stderr}"
            )
            return False

        return True

    def _kill(self) -> None:
        """Cancel the SLURM job."""
        if self._slurm_job_id is not None:
            try:
                subprocess.run(
                    ["scancel", self._slurm_job_id],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info(f"Cancelled SLURM job {self._slurm_job_id}")
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Failed to cancel SLURM job {self._slurm_job_id}: {e.stderr}"
                )

        self._slurm_job_id = None
        self._all_hostnames.clear()
