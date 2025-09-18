/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;

pub(crate) trait BorrowMeshRef<'a> {
    type Borrowed;

    fn borrow(&'a self) -> MeshRefBorrow<'a, Self::Borrowed> {
        self.try_borrow().unwrap()
    }

    fn try_borrow(&'a self) -> Result<MeshRefBorrow<'a, Self::Borrowed>, anyhow::Error>;
}

/// Multiple mesh types have two underlying implementations, one that is an owned mesh
/// and one that is a mesh ref. This enum (together with the BorrowMeshRef trait and
/// Deref impl) provides a unified way to get &MeshRef from either and owned mesh
/// or a mesh ref.
pub(crate) enum MeshRefBorrow<'a, M> {
    Owned(M),
    Ref(&'a M),
}

impl<M> MeshRefBorrow<'_, M> {
    fn borrow(&self) -> &M {
        match self {
            MeshRefBorrow::Owned(ref_) => ref_,
            MeshRefBorrow::Ref(ref_) => ref_,
        }
    }
}

impl<M> Deref for MeshRefBorrow<'_, M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}
