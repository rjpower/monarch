# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import argparse
import subprocess

from monarch._src.actor.debugger import _get_debug_server_host, _get_debug_server_port


def run():
    parser = argparse.ArgumentParser(description="Monarch Debug CLI")
    parser.add_argument(
        "--host",
        type=str,
        default=_get_debug_server_host(),
        help="Hostname where the debug server is running",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=_get_debug_server_port(),
        help="Port that the debug server is listening on",
    )
    args = parser.parse_args()

    subprocess.run(["nc", f"{args.host}", f"{args.port}"], check=True)
