# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Type hints for the monarch_hyperactor.config Rust bindings.
"""

from typing import Generic, TypeVar

from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport

def reload_config_from_env() -> None:
    """
    Reload configuration from environment variables.

    This reads all HYPERACTOR_* environment variables and updates
    the global configuration.
    """
    ...

T = TypeVar("T")

# ConfigKey isn't actually a class that exists,
# and the rust configuration keys like DefaultTransport
# don't share a common subclass. But this is nice for
# type-checking and not having to stub out get and set
# methods for every config key.
class ConfigKey(Generic[T]):
    @staticmethod
    def get() -> T: ...
    @staticmethod
    def set(val: T) -> None: ...

class DefaultTransport(ConfigKey[ChannelTransport]): ...
