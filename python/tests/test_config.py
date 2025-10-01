# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import DefaultTransport


def test_get_set_transport() -> None:
    DefaultTransport.set(ChannelTransport.Tcp)
    assert DefaultTransport.get() == ChannelTransport.Tcp
    DefaultTransport.set(ChannelTransport.MetaTlsWithHostname)
    assert DefaultTransport.get() == ChannelTransport.MetaTlsWithHostname
    DefaultTransport.set(ChannelTransport.MetaTlsWithIpV6)
    assert DefaultTransport.get() == ChannelTransport.MetaTlsWithIpV6
