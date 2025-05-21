# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchX component definition for Monarch server.
Basically a job definition for creating a Monarch server.

See: https://pytorch.org/torchx/main/basics.html#components
"""

# pyre-strict
import getpass
from typing import Dict, List, Optional

import torchx.components.fb.conda_transforms as conda_transforms
import torchx.specs as specs
from monarch.tools.mesh_spec import (
    DEFAULT_REMOTE_ALLOCATOR_PORT,
    mesh_spec_from_str,
    tag_as_metadata,
)
from torchx.specs.fb.component_helpers import fbpkg, Packages, run_as, ttls

_TAGS = ["monarch"]

_IGNORED = 1

_DEFAULT_MESHES = ["mesh_0:1:gtt_any"]

_EMPTY_PACKAGES = Packages()

_USER: str = getpass.getuser()


def hyperactor(
    name: str = f"monarch-{_USER}",
    # TODO kiuk@ figure out a better way to pass mesh specs
    #  right now TorchX component function signature is limited to
    #  primitives and list, dict of primitives so we encode
    #  mesh specs as ':' delimited str for now
    meshes: List[str] = _DEFAULT_MESHES,
    env: Optional[Dict[str, str]] = None,
    port: int = DEFAULT_REMOTE_ALLOCATOR_PORT,
    hyperactor_fbpkg: str = "monarch:prod",
    additional_packages: Packages = _EMPTY_PACKAGES,
    pre_launch_cmd: Optional[str] = None,
    # TODO kiuk@ [3/n][monarch] create run_monarch_mesh_worker.sh and hook it up here
    program: str = f"{specs.macros.img_root}/projects/monarch/scripts/run_monarch_bootstrap.sh",
    systemd_services: Optional[Dict[str, bool]] = None,
) -> specs.AppDef:
    """Creates a Monarch server on hyperactor per the given parameters.

    Args:
        meshes: list of mesh specs of the form "{name}:{num_hosts}:{host_type}"
    """

    systemd_services = systemd_services or {}
    env = env or {}

    packages = additional_packages
    packages.add_python_lib(hyperactor_fbpkg)

    launch_cmd = f"/packages/{fbpkg.from_id(hyperactor_fbpkg).name}/hyperactor"
    # TODO kiuk@: refactor pre_launch_cmd and && + ; into ChainedEntrypoint in torchx/specs/fb/component_helpers.py
    entrypoint = (
        launch_cmd if pre_launch_cmd is None else f"{pre_launch_cmd} && {launch_cmd}"
    )

    appdef = specs.AppDef(name=name)

    for mesh in [mesh_spec_from_str(mesh) for mesh in meshes]:
        mesh_role = specs.Role(
            name=mesh.name,
            image=packages.image,
            entrypoint=entrypoint,
            args=[
                # not needed for mesh-worker subcmd but is a global arg
                # TODO kiuk@: extract mesh-worker subcmd into its own entrypoint
                f"--num-hosts={_IGNORED}",
                "mesh-worker",
                f"--port={port}",
                f"--program={program}",
            ],
            num_replicas=mesh.num_hosts,
            resource=specs.resource(h=mesh.host_type),
            env=env,
            port_map={"mesh": port},
        )
        ttls(mesh_role, enable=True)
        run_as(mesh_role, root_user=True)
        tag_as_metadata(mesh, appdef)

        appdef.roles.append(mesh_role)

    conda_transforms.append_tb_logdir_metadata(appdef)
    appdef.metadata["tags"] = ",".join(_TAGS)
    return appdef
