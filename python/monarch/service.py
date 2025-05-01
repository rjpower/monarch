import asyncio
import contextvars
import ctypes
import inspect

import itertools
import random
import sys
import traceback
from asyncio import Future

from dataclasses import dataclass
from functools import cache
from traceback import extract_tb, StackSummary
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    cast,
    Concatenate,
    Coroutine,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    ParamSpec,
    Tuple,
    Type,
    TypeVar,
)

import monarch._monarch.hyperactor as hyperactor
import torch

from monarch._monarch.ndslice import Slice as NDSlice
from monarch.common.mesh_trait import MeshTrait
from monarch.common.pickle_flatten import flatten, unflatten
from monarch.python_local_mesh import _local_device_count


Allocator = hyperactor.ProcessAllocator | hyperactor.LocalAllocator

try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
except ImportError:
    IN_PAR = False

T1 = TypeVar("T1")
T2 = TypeVar("T2")


@dataclass
class MonarchContext:
    mailbox: hyperactor.Mailbox
    proc_id: str
    rank: int
    shape: hyperactor.Shape

    @staticmethod
    def get() -> "MonarchContext":
        return _context.get()


_context: contextvars.ContextVar[MonarchContext] = contextvars.ContextVar(
    "monarch.service._context"
)


# this was implemented in python 3.12 as an argument to task
# but I have to backport to 3.10/3.11.
# pyre-ignore[2]: Parameter `coro` must have a type that does not contain `Any`.
def create_eager_task(coro: Coroutine[Any, None, Any]) -> asyncio.Future:
    iter = coro.__await__()
    try:
        first_yield = next(iter)
        return asyncio.create_task(RestOfCoroutine(first_yield, iter).run())
    except StopIteration as e:
        t = asyncio.Future()
        t.set_result(e.value)
        return t


class RestOfCoroutine(Generic[T1, T2]):
    def __init__(self, first_yield: T1, iter: Generator[T2, None, T2]) -> None:
        self.first_yield: T1 | None = first_yield
        self.iter: Generator[T2, None, T2] = iter

    def __await__(self) -> Generator[T1, None, T1] | Generator[T2, None, T2]:
        first_yield = self.first_yield
        assert first_yield is not None
        yield first_yield
        self.first_yield = None
        while True:
            try:
                yield next(self.iter)
            except StopIteration as e:
                return e.value

    async def run(self) -> T1 | T2:
        return await self


T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
A = TypeVar("A")

# keep this load balancing deterministic, but
# equally distributed.
_load_balancing_seed = random.Random(4)


# standin class for whatever is the serializable python object we use
# to name an actor mesh. Hacked up today because ActorMesh
# isn't plumbed to non-clients
class ActorMeshRef:
    def __init__(
        self,
        mailbox: hyperactor.Mailbox,
        hy_actor_mesh: Optional[hyperactor.PythonActorMesh],
        shape: hyperactor.Shape,
        actor_ids: List[hyperactor.ActorId],
    ) -> None:
        self._mailbox = mailbox
        self._actor_mesh = hy_actor_mesh
        self._shape = shape
        self._please_replace_me_actor_ids = actor_ids

    @staticmethod
    def from_hyperactor_mesh(
        mailbox: hyperactor.Mailbox, hy_actor_mesh: hyperactor.PythonActorMesh
    ) -> "ActorMeshRef":
        shape: hyperactor.Shape = hy_actor_mesh.shape
        return ActorMeshRef(
            mailbox,
            hy_actor_mesh,
            hy_actor_mesh.shape,
            [
                cast(hyperactor.ActorId, hy_actor_mesh.get(i))
                for i in range(len(shape.ndslice))
            ],
        )

    @staticmethod
    def from_actor_id(
        mailbox: hyperactor.Mailbox, actor_id: hyperactor.ActorId
    ) -> "ActorMeshRef":
        return ActorMeshRef(mailbox, None, singleton_shape, [actor_id])

    @staticmethod
    def from_actor_ref_with_shape(
        ref: "ActorMeshRef", shape: hyperactor.Shape
    ) -> "ActorMeshRef":
        return ActorMeshRef(ref._mailbox, None, shape, ref._please_replace_me_actor_ids)

    def __getstate__(
        self,
    ) -> Tuple[hyperactor.Shape, List[hyperactor.ActorId], hyperactor.Mailbox]:
        return self._shape, self._please_replace_me_actor_ids, self._mailbox

    def __setstate__(
        self,
        state: Tuple[hyperactor.Shape, List[hyperactor.ActorId], hyperactor.Mailbox],
    ) -> None:
        self._actor_mesh = None
        self._shape, self._please_replace_me_actor_ids, self._mailbox = state

    def choose(self, message: hyperactor.PythonMessage) -> None:
        idx = _load_balancing_seed.randrange(len(self._shape.ndslice))
        actor_rank = self._shape.ndslice[idx]
        self._mailbox.post(self._please_replace_me_actor_ids[actor_rank], message)

    def send(self, rank: int, message: hyperactor.PythonMessage) -> None:
        actor = self._please_replace_me_actor_ids[rank]
        self._mailbox.post(actor, message)

    def cast(self, message: hyperactor.PythonMessage) -> None:
        if self._actor_mesh is None:
            # replace me with actual remote actor mesh
            for rank in self._shape.ranks():
                self._mailbox.post(self._please_replace_me_actor_ids[rank], message)
        else:
            self._actor_mesh.cast(message)

    @property
    def len(self) -> int:
        return len(self._shape.ndslice)


class ProcMesh:
    def __init__(self, hy_proc_mesh: hyperactor.ProcMesh) -> None:
        self._proc_mesh = hy_proc_mesh
        self._mailbox: hyperactor.Mailbox = self._proc_mesh.client
        self._rdma_manager_awaitable: Awaitable[RDMAManager] = self.spawn(
            "rdma_manager", RDMAManager
        )

    async def spawn(self, name: str, Class: Type[T], *args: Any, **kwargs: Any) -> T:
        if not issubclass(Class, Actor):
            raise ValueError(
                f"{Class} must subclass monarch.service.Actor to spawn it."
            )
        # init isn't async but we do not need the rdma_manager initialized until
        # we spawn something else. When there is a distinction between the client.
        if self._rdma_manager_awaitable is not None:
            self._rdma_manager_awaitable, awaitable = None, self._rdma_manager_awaitable
            await awaitable
        actor_mesh = await self._proc_mesh.spawn(name, _Actor)
        service = Service(
            Class,
            ActorMeshRef.from_hyperactor_mesh(self._mailbox, actor_mesh),
            self._mailbox,
        )
        # useful to have this separate, because eventually we can reconstitute Service objects across pickling by
        # doing `Service(Class, actor_handle)` but not calling _create.
        service._create(args, kwargs)
        return cast(T, service)


init_asyncio_loop: Any = cache(hyperactor.init_asyncio_loop)


async def local_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> ProcMesh:
    init_asyncio_loop()
    if gpus is None:
        gpus = _local_device_count()
    spec = hyperactor.AllocSpec(gpus=gpus, hosts=hosts)
    alloc = await hyperactor.LocalAllocator.allocate(spec)
    return ProcMesh(await hyperactor.ProcMesh.allocate(alloc))


_BOOTSTRAP_MAIN = "monarch._monarch.hyperactor.bootstrap_main"


def _get_bootstrap_args() -> tuple[str, Optional[list[str]], dict[str, str]]:
    if IN_PAR:
        cmd = sys.argv[0]
        args = None
        env = {
            "PAR_MAIN_OVERRIDE": _BOOTSTRAP_MAIN,
        }
    else:
        cmd = sys.executable
        args = ["-m", _BOOTSTRAP_MAIN]
        env = {}

    return cmd, args, env


async def proc_mesh(
    *, gpus: Optional[int] = None, hosts: int = 1, env: Optional[dict[str, str]] = None
) -> ProcMesh:
    init_asyncio_loop()
    if gpus is None:
        gpus = _local_device_count()
    spec = hyperactor.AllocSpec(gpus=gpus, hosts=hosts)
    env = env or {}
    cmd, args, base_env = _get_bootstrap_args()
    env.update(base_env)
    env["HYPERACTOR_MANAGED_SUBPROCESS"] = "1"
    allocator = hyperactor.ProcessAllocator(cmd, args, env)
    alloc = await allocator.allocate(spec)
    return ProcMesh(await hyperactor.ProcMesh.allocate(alloc))


class Endpoint(Generic[P, R]):
    def __init__(
        self,
        actor_mesh_ref: ActorMeshRef,
        name: str,
        impl: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
        mailbox: hyperactor.Mailbox,
    ) -> None:
        self._actor_mesh = actor_mesh_ref
        self._name = name
        self._signature: inspect.Signature = inspect.signature(impl)
        self._mailbox = mailbox

    def broadcast(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """
        Fire-and-forget broadcast invocation of the endpoint across all actors in the mesh.

        This sends the message to all actors but does not wait for any result.
        """
        self._signature.bind(None, *args, **kwargs)
        self._actor_mesh.cast(self._message(args, kwargs, None))

    def _port(
        self, once: bool
    ) -> Tuple["Port", hyperactor.OncePortReceiver | hyperactor.PortReceiver]:
        handle, receiver = (
            self._mailbox.open_once_port() if once else self._mailbox.open_port()
        )
        port_id: hyperactor.PortId = handle.bind()
        return Port(port_id, self._mailbox), receiver

    async def choose(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Load balanced sends a message to one chosen actor and awaits a result.

        Load balanced RPC-style entrypoint for request/response messaging.
        """
        self._signature.bind(None, *args, **kwargs)
        port, receiver = self._port(once=True)
        self._actor_mesh.choose(self._message(args, kwargs, port))
        return self._unpack(await receiver.recv())

    def call(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]:
        if self._actor_mesh.len != 1:
            raise ValueError(
                f"Can only use 'call' on a single Actor but this actor has shape {self._actor_mesh._shape}"
            )
        return self.choose(*args, **kwargs)

    async def stream(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[R, R]:
        """
        Broadcasts to all actors and yields their responses as a stream / generator.

        This enables processing results from multiple actors incrementally as
        they become available. Returns an async generator of response values.
        """
        self._signature.bind(None, *args, **kwargs)
        port, receiver = self._port(once=False)
        self._actor_mesh.cast(self._message(args, kwargs, port))
        for _ in range(self._actor_mesh.len):
            yield self._unpack(await receiver.recv())

    async def aggregate(
        self,
        identity: A,
        combine: Callable[[A, R], A],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> A:
        value = identity
        async for x in self.stream(*args, **kwargs):
            value = combine(value, x)
        return value

    def broadcast_and_wait(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> Coroutine[None, Any, Any]:
        """
        Broadcast to all actors and wait for each to acknowledge receipt.

        This behaves like `cast`, but ensures that each actor has received and
        processed the message by awaiting a response from each one. Does not
        return any results.
        """
        return self.aggregate(None, lambda x, _: x, *args, **kwargs)

    def _unpack(self, msg: hyperactor.PythonMessage) -> R:
        # TODO: Try to do something more structured than a cast here
        payload = cast(R, _unpickle(msg.message, self._mailbox))
        if msg.method == "result":
            return payload
        else:
            assert msg.method == "exception"
            # pyre-ignore do something more structured here
            raise payload

    def _message(
        self, args: object, kwargs: object, port: Optional["Port"]
    ) -> hyperactor.PythonMessage:
        return hyperactor.PythonMessage(self._name, _pickle((args, kwargs, port)))

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Monarch service endpoints cannot be called directly, use .call suffix for getting a result or the .cast suffix to send without expecting a result."
        )


class EndpointProperty(Generic[P, R]):
    def __init__(self, method: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]):
        self._method = method

    def __get__(self, instance, owner) -> Endpoint[P, R]:
        # this is a total lie, but we have to actually
        # recognize this was defined as an endpoint,
        # and also lookup the method
        return cast(Endpoint[P, R], self)


def endpoint(
    method: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> EndpointProperty[P, R]:
    if not inspect.iscoroutinefunction(method):
        raise TypeError(f"The implementation of an endpoint must be an async function")
    return EndpointProperty(method)


class Port:
    def __init__(self, port: hyperactor.PortId, mailbox: hyperactor.Mailbox) -> None:
        self._port = port
        self._mailbox = mailbox

    def send(self, method: str, obj: object) -> None:
        self._mailbox.post(
            self._port,
            hyperactor.PythonMessage(method, _pickle(obj)),
        )


singleton_shape = hyperactor.Shape([], NDSlice(offset=0, sizes=[], strides=[]))


class _Actor:
    def __init__(self) -> None:
        self.instance: object | None = None
        self.active_requests: asyncio.Queue[Future[object]] = asyncio.Queue()
        self.complete_task: object | None = None

    async def handle(
        self, mailbox: hyperactor.Mailbox, message: hyperactor.PythonMessage
    ) -> None:
        return await self.handle_cast(mailbox, 0, singleton_shape, message)

    async def handle_cast(
        self,
        mailbox: hyperactor.Mailbox,
        rank: int,
        shape: hyperactor.Shape,
        message: hyperactor.PythonMessage,
    ) -> None:
        try:
            _context.set(MonarchContext(mailbox, mailbox.actor_id.proc_id, rank, shape))

            args, kwargs, port = _unpickle(message.message, mailbox)
            if message.method == "__init__":
                # pyre-ignore Incompatible variable type [9]: Unable to unpack `List[typing.Callable[..., ...]]`, expected a tuple.
                Class, *args = args
                self.instance = Class(*args, **kwargs)
            else:

                async def run() -> None:
                    try:
                        result = await getattr(self.instance, message.method)._method(
                            self.instance, *args, **kwargs
                        )
                        if port is not None:
                            port.send("result", result)
                    except Exception as e:
                        s = ServiceCallFailedException(e)
                        if port is not None:
                            port.send("exception", s)
                        raise s from None

                if self.complete_task is None:
                    asyncio.create_task(self._complete())
                await self.active_requests.put(create_eager_task(run()))
        except Exception as e:
            raise ServiceCallFailedException(e) from None

    async def _complete(self) -> None:
        while True:
            task = await self.active_requests.get()
            await task


def _is_mailbox(x: object) -> bool:
    return isinstance(x, hyperactor.Mailbox)


def _pickle(obj: object) -> bytes:
    _, msg = flatten(obj, _is_mailbox)
    return msg


def _unpickle(data: bytes, mailbox: hyperactor.Mailbox) -> Any:
    # regardless of the mailboxes of the remote objects
    # they all become the local mailbox.
    return unflatten(data, itertools.repeat(mailbox))


class Actor(MeshTrait):
    @property
    def _ndslice(self) -> NDSlice:
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )

    @property
    def _labels(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )

    def _new_with_shape(self, labels: Tuple[str, ...], ndslice: NDSlice) -> "Service":
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )


class Service(MeshTrait):
    def __init__(
        self, Class: Type[T], actor_mesh_ref: ActorMeshRef, mailbox: hyperactor.Mailbox
    ) -> None:
        self._class = Class
        self._actor_mesh_ref = actor_mesh_ref
        self._mailbox = mailbox
        for attr_name in dir(self._class):
            attr_value = getattr(self._class, attr_name, None)
            if isinstance(attr_value, EndpointProperty):
                setattr(
                    self,
                    attr_name,
                    Endpoint(
                        self._actor_mesh_ref,
                        attr_name,
                        attr_value._method,
                        self._mailbox,
                    ),
                )

    def _create(self, args: Iterable[Any], kwargs: Dict[str, Any]) -> None:
        async def null_func(*_args: Iterable[Any], **_kwargs: Dict[str, Any]) -> None:
            return None

        ep = Endpoint(
            self._actor_mesh_ref,
            "__init__",
            null_func,
            self._mailbox,
        )
        # pyre-ignore
        ep.broadcast(self._class, *args, **kwargs)

    def __reduce_ex__(self, protocol: ...) -> "Tuple[Type[Service], Tuple[Any, ...]]":
        return Service, (
            self._class,
            self._actor_mesh_ref,
            self._mailbox,
        )

    @property
    def _ndslice(self) -> NDSlice:
        return self._actor_mesh_ref._shape.ndslice

    @property
    def _labels(self) -> Iterable[str]:
        return self._actor_mesh_ref._shape.labels

    def _new_with_shape(self, labels: Tuple[str, ...], ndslice: NDSlice) -> "Service":
        return Service(
            self._class,
            ActorMeshRef.from_actor_ref_with_shape(
                self._actor_mesh_ref, hyperactor.Shape(list(labels), ndslice)
            ),
            self._mailbox,
        )


@dataclass
class LocalRDMARecord:
    data: torch.ByteTensor


_local_buffers: Dict[int, "LocalRDMARecord"] = {}


def _get_bytes(storage: torch.ByteTensor, offset: int, size: int) -> bytearray:
    if offset + size > storage.numel():
        raise ValueError(f"Read out of range: {offset + size} > {storage.size()}")
    addr = storage.data_ptr()
    if storage.device.type != "cpu":
        result = bytearray(size)
        result_tensor = torch.frombuffer(
            result,
            dtype=torch.uint8,
        )
        source_tensor = storage[offset:]
        result_tensor.copy_(source_tensor)
    else:
        ctypes_array = (ctypes.c_byte * size).from_address(addr)
        result = bytearray(ctypes_array)
    return result


class RDMAManager(Actor):
    @staticmethod
    def on_proc(proc_id: str) -> "RDMAManager":
        ctx = MonarchContext.get()
        return cast(
            RDMAManager,
            Service(
                RDMAManager,
                ActorMeshRef.from_actor_id(
                    ctx.mailbox,
                    hyperactor.ActorId.from_string(f"{proc_id}.rdma_manager[0]"),
                ),
                ctx.mailbox,
            ),
        )

    @endpoint
    async def drop(self, addr: int) -> None:
        if addr in _local_buffers:
            del _local_buffers[addr]

    @endpoint
    async def fetch(self, addr: int, offset: int, nbytes: int) -> bytearray:
        if addr not in _local_buffers:
            raise ValueError(f"Unknown buffer {addr}")
        storage = _local_buffers[addr].data
        return _get_bytes(storage, offset, nbytes)

    @endpoint
    async def put(self, addr: int, offset: int, bytes: bytearray) -> None:
        if addr not in _local_buffers:
            raise ValueError(f"Unknown buffer {addr}")
        storage = _local_buffers[addr].data
        storage[offset : offset + len(bytes)] = torch.frombuffer(
            bytes, dtype=storage.dtype
        )


class RDMABuffer:
    def __init__(self, data: torch.Tensor) -> None:
        if data._is_view():
            raise ValueError("data buffer must not be a view of another tensor.")
        assert data.storage_offset() == 0
        assert data.is_contiguous()
        storage = data.untyped_storage()
        self.addr: int = storage.data_ptr()
        self.begin = 0
        self.end: int = storage.size()
        self.proc_id: str = MonarchContext.get().proc_id
        self.local_data: object = None
        _local_buffers[self.addr] = LocalRDMARecord(torch.ByteTensor(storage))

    def drop(self) -> None:
        if self.proc_id is None:
            del _local_buffers[self.addr]
            return
        rmda_actor = RDMAManager.on_proc(self.proc_id)
        # pyre-ignore[16]: Undefined attribute [16]: `Endpoint` has no attribute `cast`.
        rmda_actor.drop.cast(self.addr)

    def __getstate__(self) -> Tuple[int, int, int, Optional[str]]:
        proc_id = self.proc_id
        # locally created RDMABuffer being set remotely,
        # record its proc_id so we know how to establish connections to it
        if proc_id is None:
            proc_id = MonarchContext.get().proc_id
        return (self.addr, self.begin, self.end, proc_id)

    def __setstate__(self, state: Tuple[int, int, int, str]) -> None:
        self.local_data = None
        self.addr, self.begin, self.end, self.proc_id = state

    async def read_into(self, dst: torch.Tensor, offset: int = 0) -> None:
        if not dst.is_contiguous():
            raise ValueError("destination must be contiguous")
        nbytes = dst.element_size() * dst.numel()
        bytes = await RDMAManager.on_proc(self.proc_id).fetch.call(
            self.addr, offset, nbytes
        )
        torch.ByteTensor(dst.untyped_storage()).copy_(
            torch.frombuffer(bytes, dtype=torch.uint8)
        )

    async def write(self, src: torch.Tensor, offset: int = 0) -> None:
        if not src.is_contiguous():
            raise ValueError("destination must be contiguous")
        nbytes = src.element_size() * src.numel()
        bytes = _get_bytes(
            torch.ByteTensor(src.untyped_storage()),
            cast(int, src.storage_offset()),
            nbytes,
        )
        await RDMAManager.on_proc(self.proc_id).put.call(self.addr, offset, bytes)


class ServiceCallFailedException(Exception):
    """
    Deterministic problem with the user's code.
    For example, an OOM resulting in trying to allocate too much GPU memory, or violating
    some invariant enforced by the various APIs.
    """

    def __init__(
        self,
        exception: Exception,
        message: str = "A remote service call has failed asynchronously.",
    ) -> None:
        self.exception = exception
        self.service_frames: StackSummary = extract_tb(exception.__traceback__)
        self.message = message

    def __str__(self) -> str:
        exe = str(self.exception)
        service_tb = "".join(traceback.format_list(self.service_frames))
        return (
            f"{self.message}\n"
            f"Traceback of where the service call failed (most recent call last):\n{service_tb}{type(self.exception).__name__}: {exe}"
        )


def current_actor_name() -> str:
    return str(MonarchContext.get().mailbox.actor_id)


def current_rank() -> Dict[str, int]:
    ctx = MonarchContext.get()
    return ctx.shape.coordinates(ctx.rank)


def current_size() -> Dict[str, int]:
    ctx = MonarchContext.get()
    return dict(zip(ctx.shape.labels, ctx.shape.ndslice.sizes))
