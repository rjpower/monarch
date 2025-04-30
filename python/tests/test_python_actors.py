import asyncio
import operator
from typing import Awaitable

import torch
from monarch.service import (
    Actor,
    current_actor_name,
    current_rank,
    current_size,
    endpoint,
    local_proc_mesh,
    proc_mesh,
    RDMABuffer,
)


class Counter(Actor):
    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        self.v += 1

    @endpoint
    async def value(self) -> int:
        return self.v


class Indirect(Actor):
    @endpoint
    async def call_value(self, c: Counter) -> int:
        return await c.value.choose()


# have to use a single loop for all tests, otherwise there are
# loop closed errors.

loop = asyncio.get_event_loop()


def run_async(x):
    return lambda: loop.run_until_complete(x())


class ParameterServer(Actor):
    def __init__(self):
        self.params = torch.rand(10, 10)
        self.grad_buffer = torch.rand(10, 10)

    @endpoint
    async def grad_handle(self) -> RDMABuffer:
        return RDMABuffer(self.grad_buffer)

    async def update(self):
        self.params += 0.01 * self.grad_buffer

    async def log(self):
        print(self.params)


@run_async
async def test_choose():
    proc = await local_proc_mesh(gpus=2)
    v = await proc.spawn("counter", Counter, 3)
    i = await proc.spawn("indirect", Indirect)
    v.incr.broadcast()
    result = await v.value.choose()
    result2 = await i.call_value.choose(v)

    assert result == result2


@run_async
async def test_stream():
    proc = await local_proc_mesh(gpus=2)
    v = await proc.spawn("counter2", Counter, 3)
    v.incr.broadcast()

    assert 8 == sum([x async for x in v.value.stream()])


class ParameterClient(Actor):
    def __init__(self, server, buffer):
        self.server = server
        self.buffer = buffer

    @endpoint
    async def upload(self, tensor):
        gh = await self.server.grad_handle.call()
        await gh.write(tensor)

    @endpoint
    async def download(self):
        gh = await self.server.grad_handle.call()
        await gh.read_into(self.buffer)

    @endpoint
    async def get_buffer(self):
        return self.buffer


@run_async
async def test_rdma():
    proc = await local_proc_mesh(gpus=1)
    server = await proc.spawn("server", ParameterServer)
    client = await proc.spawn("client", ParameterClient, server, torch.ones(10, 10))

    buffer = await client.get_buffer.call()
    assert torch.sum(buffer) == 100

    tensor = torch.zeros(10, 10)

    await client.upload.call(tensor)
    await client.download.call()

    buffer = await client.get_buffer.call()
    assert torch.sum(buffer) == 0


@run_async
async def test_proc_mesh_rdma():
    proc = await proc_mesh(gpus=1)
    server = await proc.spawn("server", ParameterServer)
    client = await proc.spawn("client", ParameterClient, server, torch.ones(10, 10))

    buffer = await client.get_buffer.call()
    assert torch.sum(buffer) == 100

    tensor = torch.zeros(10, 10)

    await client.upload.call(tensor)
    await client.download.call()

    buffer = await client.get_buffer.call()
    assert torch.sum(buffer) == 0


class To(Actor):
    @endpoint
    async def whoami(self):
        return current_actor_name()


class From(Actor):
    @endpoint
    async def get(self, to: To):
        return [x async for x in to.whoami.stream()]


@run_async
async def test_mesh_passed_to_mesh():
    proc = await local_proc_mesh(gpus=2)
    f = await proc.spawn("from", From)
    t = await proc.spawn("to", To)
    all = [y async for x in f.get.stream(t) for y in x]
    assert len(all) == 4
    assert all[0] != all[1]


@run_async
async def test_mesh_passed_to_mesh_on_different_proc_mesh():
    proc = await local_proc_mesh(gpus=2)
    proc2 = await local_proc_mesh(gpus=2)
    f = await proc.spawn("from", From)
    t = await proc2.spawn("to", To)
    all = [y async for x in f.get.stream(t) for y in x]
    assert len(all) == 4
    assert all[0] != all[1]


@run_async
async def test_actor_slicing():
    proc = await local_proc_mesh(gpus=2)
    proc2 = await local_proc_mesh(gpus=2)

    f = await proc.spawn("from", From)
    t = await proc2.spawn("to", To)

    assert await t.slice(gpus=0).whoami.call() != await t.slice(gpus=1).whoami.call()

    result = [y async for x in f.get.stream(t.slice(gpus=0)) for y in x]
    assert len(result) == 2

    assert result[0] == result[1]


@run_async
async def test_aggregate():
    proc = await local_proc_mesh(gpus=2)
    counter = await proc.spawn("counter", Counter, 1)
    counter.incr.broadcast()
    r = await counter.value.aggregate(0, operator.add)
    assert r == 4


class RunIt(Actor):
    @endpoint
    async def run(self, fn):
        return fn()


@run_async
async def test_rank_size():
    proc = await local_proc_mesh(gpus=2)
    r = await proc.spawn("runit", RunIt)

    assert 1 == await r.run.aggregate(0, operator.add, lambda: current_rank()["gpus"])
    assert 4 == await r.run.aggregate(0, operator.add, lambda: current_size()["gpus"])
