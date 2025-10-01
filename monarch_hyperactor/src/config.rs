/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Configuration for Monarch Hyperactor.
//!
//! This module provides monarch-specific configuration attributes that extend
//! the base hyperactor configuration system.

use hyperactor::attrs::declare_attrs;
use hyperactor_mesh::proc_mesh::DEFAULT_TRANSPORT;
use pyo3::prelude::*;

use crate::channel::PyChannelTransport;

// Declare monarch-specific configuration keys
declare_attrs! {
    /// Use a single asyncio runtime for all Python actors, rather than one per actor
    pub attr SHARED_ASYNCIO_RUNTIME: bool = false;
}

/// Python API for configuration management
///
/// Reload configuration from environment variables
#[pyfunction()]
pub fn reload_config_from_env() -> PyResult<()> {
    // Reload the hyperactor global configuration from environment variables
    hyperactor::config::global::init_from_env();
    Ok(())
}

struct ConfigKeyInfo {
    register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
}

inventory::collect!(ConfigKeyInfo);

macro_rules! register_config_key {
    ($id:ident) => {
        hyperactor::paste! {
            hyperactor::submit! {
                ConfigKeyInfo {
                    register: |module| {
                        module.add_class::<[<PY_ $id>]>()
                    }
                }
            }
        }
    };
}

fn _on_set() -> PyResult<()> {
    Ok(())
}

/// Define python bindings to make
macro_rules! py_configurable {
    (py, $id:ident, $py_name:literal, $py_ty:ty) => {
        py_configurable!(py, $id, $py_name, $py_ty, _on_set);
    };
    ($id:ident, $py_name:literal, $ty:ty) => {
        py_configurable!($id, $py_name, $ty, _on_set);
    };
    (py, $id:ident, $py_name:literal, $py_ty:ty, $on_set:ident) => {
        hyperactor::paste! {
            #[pyclass(name = $py_name, module = "monarch._rust_bindings.monarch_hyperactor.config", frozen)]
            #[allow(non_camel_case_types)]
            #[derive(Clone)]
            struct [<PY_ $id>];

            #[pymethods]
            impl [<PY_ $id>] {
                #[staticmethod]
                fn get() -> PyResult<Option<$py_ty>> {
                    hyperactor::config::global::try_get_cloned($id)
                        .map(|val| val.try_into())
                        .transpose()
                }

                #[staticmethod]
                fn set(val: &$py_ty) -> PyResult<()> {
                    hyperactor::config::global::set($id, val.clone().into());
                    $on_set()
                }
            }

            register_config_key!($id);
        }
    };
    ($id:ident, $py_name:literal, $ty:ty, $on_set:expr) => {
        hyperactor::paste! {
            #[pyclass(name = $py_name, module = "monarch._rust_bindings.monarch_hyperactor.config", frozen)]
            #[allow(non_camel_case_types)]
            #[derive(Clone)]
            struct [<PY_ $id>];

            #[pymethods]
            impl [<PY_ $id>] {
                #[staticmethod]
                fn get() -> Option<$ty> {
                    hyperactor::config::global::try_get_cloned($id)
                }

                #[staticmethod]
                fn set(val: $ty) -> PyResult<()> {
                    hyperactor::config::global::set($id, val);
                    $on_set()
                }
            }

            register_config_key!($id);
        }
    };
}

// TODO(slurye): Add a callback to re-initialize the root client
// when default transport changes.
py_configurable!(
    py,
    DEFAULT_TRANSPORT,
    "DefaultTransport",
    PyChannelTransport
);

/// Register Python bindings for the config module
pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let reload = wrap_pyfunction!(reload_config_from_env, module)?;
    reload.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_hyperactor.config",
    )?;
    module.add_function(reload)?;

    for key in inventory::iter::<ConfigKeyInfo>() {
        (key.register)(module)?;
    }

    Ok(())
}
