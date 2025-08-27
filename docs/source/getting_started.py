# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monarch Getting Started Guide
=============================

This guide introduces the core concepts of Monarch, a framework for building
multi-machine training programs using actors. We'll cover:

- Defining actors with endpoint functions
- Spawning actors locally and across multiple hosts
- Sending messages and organizing actors into meshes
- The supervision tree for fault tolerance
- Distributed tensors and RDMA capabilities
"""

from monarch._src.actor.actor_mesh import context
# %%
# Defining an Actor
# -----------------
# At its core, Monarch uses actors as a way to create multi-machine training programs.
# Actors are Python objects that expose a number of endpoint functions. These functions
# can be called by other actors in the system and their responses gathered asynchronously.
#
# Let's start by defining a simple actor:

from monarch.actor import Actor, endpoint, Future, this_host, this_proc


class Counter(Actor):
    def __init__(self, initial_value: int):
        self.value = initial_value

    @endpoint
    def increment(self) -> None:
        self.value += 1

    @endpoint
    def get_value(self) -> int:
        return self.value


# The decorator `endpoint` specifies functions of the Actor that can be called remotely
# from other actors. We do it this way to ensure that IDE can offer autocompletions of
# actor calls with correct types.

# %%
# Spawning An Actor In The Local Process
# ======================================
# We can then spawn an actor in the currently running process like so:

counter: Counter = this_proc().spawn("counter", Counter, initial_value=0)

# `this_proc()` is a handle to a process and lets us directly control where an actor runs.
# Monarch is very literal about where things run so that code can be written in the most
# efficient way. For instance, two actors in the same proc can rely on the fact that they
# have the same memory space. Two actors on the same host can communicate through /dev/shm, etc.

# %%
# Sending A Simple Message
# ========================
# Once spawned we can send messages to the actor:

fut: Future[int] = counter.get_value.call_one()
value: int = fut.get()

print(f"Counter value: {value}")

# Here we invoked the get_value message, returning 0, the current value of the Counter.
# We refer to the `call_one` method as an 'adverb' because it modifies how results of the
# endpoint are handled. `call_one` just invokes a single actor and gets its value.
#
# Notice the return value is a Future[int] -- the message is sent asynchronously, letting
# the sender do other things before it needs the reply. `get()` waits on the result.

# %%
# Multiple Actors at Once
# =======================
# Monarch scales to thousands of machines because of its ability to broadcast a single
# message to many actors at once rather than send many point-to-point messages. By
# organizing communication into trees, monarch can multicast messages to thousands of
# machines in a scalable way with good latency.
#
# We express broadcasted communication by organizing actors into a **Mesh** -- a
# multidimensional container with named dimensions. For instance a cluster might have
# dimensions {"hosts": 32, "gpus": 8}. Dimension names are normally things like "hosts",
# indexing across the hosts in a cluster, or "gpus", indexing across things in a machine.

from monarch.actor import ProcMesh

# To create a mesh of actors, we first create a mesh of processes, and spawn an actor on them:
procs: ProcMesh = this_host().spawn_procs(per_host={"gpus": 8})
counters: Counter = procs.spawn("counters", Counter, 0)

# %%
# Broadcasting Messages
# ---------------------
# Now messages can be sent to all the actors in the mesh:

counters.increment.broadcast()

# The `broadcast` adverb means that we are sending a message to all members of the mesh,
# and then not waiting for any response.

# %%
# Slicing Meshes
# --------------
# We can also slice up the mesh and send to only some of it (gpus 0 through 4):

counters.slice(gpus=slice(0, 4)).increment.broadcast()

# The `call` adverb lets us broadcast a message to a group of actors, and then aggregate
# their responses back:
print(counters.get_value.call().get())

# `broadcast` and `call` are the most commonly used adverbs. The `call_one` adverb we used
# earlier is actually just a special case of `call`, asserting that you know there is only
# a single actor being messaged.

# %%
# Multiple Hosts
# ==============
# When we created our processes before, we spawned them on `this_host()` -- the machine
# running the top-level script. For larger jobs, monarch controls many machines. How these
# machines are obtained depends on the scheduling system (slurm, kubernetes, etc), but these
# schedulers are typically encapsulated in a config file.

# We obtain the mesh of hosts for the job by loading that config:
# hosts: HostMesh = hosts_from_config("MONARCH_HOSTS")  # NYI: hosts_from_config


# Let's imagine we are writing a trainer across multiple hosts:
class Trainer(Actor):
    @endpoint
    def step(self):
        my_point = context().message_rank()
        print(f"Trainer {my_point} taking a step.")


# trainer_procs: ProcMesh = hosts.spawn_procs(per_host={'gpus': 8})
# print(f"Process mesh extent: {trainer_procs.extent}")
# trainers = trainer_procs.spawn(Trainer)
#
# # Do one training step and wait for all to finish it
# trainers.step.call().get()

# %%
# The Supervision Tree
# ====================
# Large scale training will run into issues with faulty hardware, flaky networks, and
# software bugs. Monarch is designed to provide good behaviors for errors and faults by
# default with the option to define more sophisticated behavior for faster recovery or
# fault tolerance.
#
# We borrow from Erlang the idea of a tree of supervision. Each process, and each actor
# has a parent that launched it and is responsible for its health.


class Errorful(Actor):
    @endpoint
    def say_hello(self):
        raise Exception("I don't want to")


# If we are calling the endpoint and expecting a response, the error will get routed to the caller:
e = this_proc().spawn("errorful", Errorful)
try:
    e.say_hello.call_one()
except Exception:
    print("It didn't say hello")

# %%
# Broadcasting Without Response
# ----------------------------
# We also might call something and provide it no way to respond:

e.say_hello.broadcast()  # do not expect a response

# The default behavior of the supervision tree means that an error at any level of process
# or actor creation will not go unreported. This can prevent a lot of accidental deadlocks
# compared to the standard unix-style defaults where threads and processes exiting do not
# stop the spawning processes.

# %%
# Fine-grained Supervision
# ========================
# Sometimes fine-grained recovery is possible. For instance, if a data loader failed to
# read a URL, perhaps it would work to just restart it. In these cases, we also offer a
# different API. If an actor defines a `__supervise__` special method, then it will get
# called in response to any supervision event.


class SupervisedActor(Actor):
    def __supervise__(self, event):
        # NYI: specific supervise protocol is not specced out or implemented.
        print(f"Supervision event received: {event}")
        # Logic to handle supervision events and potentially restart failed actors


# %%
# Actor and Process References
# ============================
# Actors, processes, and hosts can be sent between actors as arguments to messages.
# - Actors can be passed to other actors as references
# - Processes can be passed as well (but NYI)


class ReferenceExample(Actor):
    @endpoint
    def store_actor_ref(self, other_actor):
        self.other_actor = other_actor

    @endpoint
    def call_stored_actor(self):
        return self.other_actor.get_value.call_one()


# %%
# Distributed Tensors
# ===================
# Monarch also comes with a 'tensor engine' that provides distributed torch tensors.
# This lets a single actor directly compute with tensors distributed across a mesh of processes.
#
# We can use distributed features by 'activating' a ProcMesh:

# with proc_mesh.activate():
#     t = torch.rand(3, 4)

# The tensor `t` is now a distributed tensor with a unique value across each process in the mesh.
# Values across processes can be computed using the `monarch.reduce` functions.
#
# A distributed tensor can be passed to any mesh of actors that is located on the same mesh
# of processes as the tensors. Each actor will receive the shard of the distributed tensor
# that is on the same process as the actor.

# %%
# Actor Context
# =============
# At any point, code might need to know 'what' it is and 'where' it is. For instance,
# when initializing torch.distributed, it will need a RANK/WORLD_SIZE. In monarch, the
# 'what' is always what actor is currently running the code.


class ContextAwareActor(Actor):
    @endpoint
    def print_context_info(self):
        actor_instance = context().actor_instance
        message_rank = context().message_rank()
        print(f"Actor rank: {actor_instance.rank}")
        print(f"Message rank: {message_rank}")


# `context().message_rank` returns the coordinates in the message. This isn't always the
# same as the actor's rank because messages can be sent to a slice of actors rather than
# the whole thing.

# %%
# Summary
# =======
# We have now seen all four major features of Monarch:
#
# 1. **Scalable messaging** using multidimensional meshes of actors
# 2. **Fault tolerance** through supervision trees and __supervise__
# 3. **Point-to-point low-level RDMA** (not demonstrated in this example)
# 4. **Built-in distributed tensors**
#
# This foundation enables building sophisticated multi-machine training programs with
# clear semantics for distribution, fault tolerance, and communication patterns.

# %%
# TODO: Point-to-Point RDMA
# =========================
# The following topics need to be documented with examples:
#
# - Use of RDMABuffer for direct memory access
# - One-sided transfers for high-performance communication
# - Using RDMA to build communication primitives
# - Examples of one-sided transfer patterns
#
# This will demonstrate Monarch's low-level RDMA capabilities for building
# high-performance distributed systems.

# TODO: Add RDMABuffer examples here

# %%
# TODO: Free Functions
# ====================
# Documentation needed for free functions in Monarch:
#
# - Functions that operate on meshes without being bound to actors

# TODO: Add free function examples here

# %%
# TODO: Ports
# ===========
# Documentation needed for Monarch's port system:
#
# - Using ports for structured communication between actors#

# %%
# TODO: Message Ordering
# ======================
# Documentation needed for message ordering guarantees:
#
# - Understanding Monarch's message ordering semantics
# - When messages are delivered in order vs out of order

# %%
# TODO: Async Actors
# ==================
# Documentation needed for asynchronous actor patterns:
#
# - Defining async endpoint methods
# - Managing async/await patterns in actor code
# - Coordinating async operations across multiple actors
# - Best practices for async actor design
#
# This will show how to use Python's async features with Monarch actors.

# TODO: Add async actor examples here

# %%
# TODO: Responding Out of Order
# =============================
# Documentation needed for out-of-order response patterns:
#
# - When and why to respond to messages out of order
# - Managing response ordering in distributed systems
# - Using futures and promises for async responses
# - Handling partial failures in distributed responses
#
# This section should cover advanced messaging patterns for performance.

# TODO: Add out-of-order response examples here

print("Getting started guide completed!")
