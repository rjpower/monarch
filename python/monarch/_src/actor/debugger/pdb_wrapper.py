# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import bdb
import inspect
import io
import linecache
import os
import pdb  # noqa
import socket
import sys
from dataclasses import dataclass

from typing import Dict, TYPE_CHECKING

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._src.actor.sync_state import fake_sync_state

if TYPE_CHECKING:
    from monarch._src.actor.debugger.debugger import DebugController


@dataclass
class DebuggerWrite:
    payload: bytes
    function: str | None
    lineno: int | None


class PdbWrapper(pdb.Pdb):
    def __init__(
        self,
        rank: int,
        coords: Dict[str, int],
        actor_id: ActorId,
        controller: "DebugController",
        header: str | None = None,
    ):
        self.rank = rank
        self.coords = coords
        self.header = header
        self.actor_id = actor_id
        self.controller = controller
        # pyre-ignore
        super().__init__(stdout=WriteWrapper(self), stdin=ReadWrapper.create(self))
        self._first = True

        for attr in dir(self):
            # Before running each command, we need to check if we have access
            # to the source code associated with the current frame (which may
            # not be the case if the code was pickled by value), and if not,
            # retrieve the source code from the remote host and populate the
            # linecache with it.
            #
            # Each pdb command has an associated method in pdb.Pdb whose name begins
            # with "do_". We can overwrite each method to begin by doing source code
            # retrieval and populating the linecache if necessary.
            if attr.startswith("do_"):

                def wrapper(fn):
                    def impl(*args, **kwargs):
                        if self.curframe is not None:
                            self._maybe_populate_linecache_from_remote(self.curframe)
                        return fn(*args, **kwargs)

                    return impl

                setattr(self, attr, wrapper(getattr(self, attr)))

    def set_trace(self, frame=None):
        with fake_sync_state():
            self.controller.debugger_session_start.call_one(
                self.rank,
                self.coords,
                socket.getfqdn(socket.gethostname()),
                self.actor_id.actor_name,
            ).get()
        if self.header:
            self.message(self.header)
        if frame is not None:
            # We may have hit a breakpoint inside code that was pickled by value,
            # so handle that case.
            self._maybe_populate_linecache_from_remote(frame)
        super().set_trace(frame)

    def do_clear(self, arg):
        if not arg:
            # Sending `clear` without any argument specified will
            # request confirmation from the user using the `input` function,
            # which bypasses our ReadWrapper and causes a hang on the client.
            # To avoid this, we just clear all breakpoints instead without
            # confirmation.
            super().clear_all_breaks()
        else:
            super().do_clear(arg)

    def end_debug_session(self):
        with fake_sync_state():
            self.controller.debugger_session_end.call_one(
                self.actor_id.actor_name, self.rank
            ).get()
        # Once the debug client actor is notified of the session being over,
        # we need to prevent any additional requests being sent for the session
        # by redirecting stdin and stdout.
        self.stdin = sys.stdin
        self.stdout = sys.stdout

    def post_mortem(self, exc_tb):
        self._first = False
        # See builtin implementation of pdb.post_mortem() for reference.
        self.reset()
        self.interaction(None, exc_tb)

    def format_stack_entry(self, frame_lineno, lprefix):
        # This method is used inside commands like `where` and `bt`.
        # If we hit a breakpoint in a function that was called from
        # code that was pickled by value, we need to make sure that
        # the source code is in the linecache for each frame in the
        # stack.
        frame, _ = frame_lineno
        self._maybe_populate_linecache_from_remote(frame)
        return super().format_stack_entry(frame_lineno, lprefix)

    def _maybe_populate_linecache_from_remote(self, frame):
        filename = os.path.abspath(frame.f_code.co_filename)
        if os.path.exists(filename):
            return
        elif filename not in linecache.cache:
            with fake_sync_state():
                # If the file associated with the current frame doesn't exist on
                # the current host, it probably means the code was pickled by value.
                # The code should live on the same machine as the debug controller,
                # so we can request it from there and populate the linecache with it.
                # Once we've done that, pdb should work normally.
                try:
                    source = self.controller.get_source.call_one(filename).get()
                    linecache.cache[filename] = (
                        len(source),
                        None,
                        source.splitlines(keepends=True),
                        filename,
                    )
                except Exception as e:
                    self.error(
                        (
                            f"Failed to get source for {filename} due to: {e}. "
                            "Debugging experience may be degraded."
                        )
                    )


class ReadWrapper(io.RawIOBase):
    def __init__(self, session: "PdbWrapper"):
        self.session = session

    def readinto(self, b):
        with fake_sync_state():
            response = self.session.controller.debugger_read.call_one(
                self.session.actor_id.actor_name, self.session.rank, len(b)
            ).get()
            if response == "detach":
                # this gets injected by the worker event loop to
                # get the worker thread to exit on an Exit command.
                raise bdb.BdbQuit
            assert isinstance(response, DebuggerWrite) and len(response.payload) <= len(
                b
            )
            b[: len(response.payload)] = response.payload
            return len(response.payload)

    def readable(self) -> bool:
        return True

    @classmethod
    def create(cls, session: "PdbWrapper"):
        return io.TextIOWrapper(io.BufferedReader(cls(session)))


class WriteWrapper:
    def __init__(self, session: "PdbWrapper"):
        self.session = session

    def writable(self) -> bool:
        return True

    def write(self, s: str):
        function = None
        lineno = None
        if self.session.curframe is not None:
            # pyre-ignore
            function = f"{inspect.getmodulename(self.session.curframe.f_code.co_filename)}.{self.session.curframe.f_code.co_name}"
            # pyre-ignore
            lineno = self.session.curframe.f_lineno
        with fake_sync_state():
            self.session.controller.debugger_write.call_one(
                self.session.actor_id.actor_name,
                self.session.rank,
                DebuggerWrite(
                    s.encode(),
                    function,
                    lineno,
                ),
            ).get()

    def flush(self):
        pass
