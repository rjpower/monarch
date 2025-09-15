/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;

use hyperactor_mesh::comm::multicast::CastInfo;
use hyperactor_mesh::proc_mesh::global_root_client;
use ndslice::Extent;
use ndslice::Point;
use pyo3::prelude::*;

use crate::actor::PythonActor;
use crate::mailbox::PyMailbox;
use crate::proc::PyActorId;
use crate::shape::PyPoint;

#[derive(Debug)]
pub(crate) enum InstanceActorType {
    Unit,
    PythonActor,
}

#[pyclass(name = "Instance", module = "monarch._src.actor.actor_mesh")]
pub(crate) struct PyInstance {
    inner: hyperactor::proc::ErasedInstance,
    // Will be used in a future diff.
    #[allow(dead_code)]
    type_: InstanceActorType,
    #[pyo3(get, set)]
    proc_mesh: Option<PyObject>,
    #[pyo3(get, set, name = "_controller_controller")]
    controller_controller: Option<PyObject>,
    #[pyo3(get, set)]
    rank: PyPoint,
    #[pyo3(get, set, name = "_children")]
    children: Option<PyObject>,
}

#[pymethods]
impl PyInstance {
    #[getter]
    fn _mailbox(&self) -> PyMailbox {
        PyMailbox {
            inner: self.inner.mailbox_for_py().clone(),
        }
    }

    #[getter]
    fn actor_id(&self) -> PyActorId {
        self.inner.self_id().clone().into()
    }
}

impl PyInstance {
    // Will be used in a future diff
    #[allow(dead_code)]
    pub(crate) fn actor_type(&self) -> &InstanceActorType {
        &self.type_
    }
}

trait ConvertibleToPyInstance: hyperactor::Actor {
    fn actor_type() -> InstanceActorType;
}

impl ConvertibleToPyInstance for () {
    fn actor_type() -> InstanceActorType {
        InstanceActorType::Unit
    }
}

impl ConvertibleToPyInstance for PythonActor {
    fn actor_type() -> InstanceActorType {
        InstanceActorType::PythonActor
    }
}

impl<A: ConvertibleToPyInstance> From<&hyperactor::Instance<A>> for PyInstance {
    fn from(ins: &hyperactor::Instance<A>) -> Self {
        PyInstance {
            inner: ins.erased_for_py(),
            type_: A::actor_type(),
            proc_mesh: None,
            controller_controller: None,
            rank: PyPoint::new(0, Extent::unity().into()),
            children: None,
        }
    }
}

impl<A: ConvertibleToPyInstance> From<&hyperactor::Context<'_, A>> for PyInstance {
    fn from(cx: &hyperactor::Context<A>) -> Self {
        PyInstance::from(cx.deref())
    }
}

#[pyclass(name = "Context", module = "monarch._src.actor.actor_mesh")]
pub(crate) struct PyContext {
    instance: Py<PyInstance>,
    rank: Point,
}

#[pymethods]
impl PyContext {
    #[getter]
    fn actor_instance(&self) -> &Py<PyInstance> {
        &self.instance
    }

    #[getter]
    fn message_rank(&self) -> PyPoint {
        self.rank.clone().into()
    }

    #[staticmethod]
    fn _root_client_context(py: Python<'_>) -> PyResult<PyContext> {
        let instance: PyInstance = global_root_client().into();
        Ok(PyContext {
            instance: instance.into_pyobject(py)?.into(),
            rank: Extent::unity().point_of_rank(0).unwrap(),
        })
    }
}

impl PyContext {
    pub(crate) fn new<T: hyperactor::actor::Actor>(
        cx: &hyperactor::Context<T>,
        instance: Py<PyInstance>,
    ) -> PyContext {
        PyContext {
            instance,
            rank: cx.cast_info(),
        }
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyInstance>()?;
    hyperactor_mod.add_class::<PyContext>()?;
    Ok(())
}
