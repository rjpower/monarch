# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import functools
import importlib.abc
import linecache
import os

from monarch._src.actor.actor_mesh import Actor
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.proc_mesh import get_or_spawn_controller
from monarch._src.actor.sync_state import fake_sync_state


class SourceLoaderController(Actor):
    @endpoint
    def get_source(self, filename: str) -> str:
        return "".join(linecache.getlines(filename))


@functools.cache
def source_loader_controller() -> SourceLoaderController:
    with fake_sync_state():
        return get_or_spawn_controller("source_loader", SourceLoaderController).get()


@functools.cache
def load_remote_source(filename: str) -> str:
    with fake_sync_state():
        return source_loader_controller().get_source.call_one(filename).get()


class RemoteImportLoader(importlib.abc.Loader):
    def __init__(self, filename: str):
        self._filename = filename

    def get_source(self, _module_name: str) -> str:
        if os.path.exists(self._filename):
            with open(self._filename, "r") as f:
                return f.read()
        return load_remote_source(self._filename)
