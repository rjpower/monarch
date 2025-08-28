# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import sys

from monarch._src.actor.debugger import _get_debug_connection


async def run():
    reader, writer = await _get_debug_connection()

    while True:
        cmd = (await reader.read(1)).decode()
        msg_len = int.from_bytes(await reader.read(4), "big")
        message = (await reader.read(msg_len)).decode()
        match cmd:
            case "o":
                sys.stdout.write(message)
                sys.stdout.flush()
            case "i":
                inp = input(message).encode()
                writer.write(len(inp).to_bytes(4, "big"))
                writer.write(inp)
                await writer.drain()
            case "q":
                writer.close()
                await writer.wait_closed()
                sys.exit(0)
