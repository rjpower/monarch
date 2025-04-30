use std::collections::HashMap;
use std::sync::Arc;

use hyperactor_mesh::Shape;
use hyperactor_mesh::alloc::AllocConstraints;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::LocalAlloc;
use hyperactor_mesh::alloc::LocalAllocator;
use hyperactor_mesh::alloc::ProcessAlloc;
use hyperactor_mesh::alloc::ProcessAllocator;
use ndslice::Slice;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyType;
use tokio::process::Command;

#[pyclass(name = "AllocSpec", module = "monarch._monarch.hyperactor")]
pub struct PyAllocSpec {
    inner: AllocSpec,
}

#[pymethods]
impl PyAllocSpec {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(kwargs) = kwargs else {
            return Err(PyValueError::new_err(
                "Shape must have at least one dimension",
            ));
        };
        let shape_dict = kwargs.downcast::<PyDict>()?;

        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (key, value) in shape_dict {
            keys.push(key.clone());
            values.push(value.clone());
        }

        let shape = Shape::new(
            keys.into_iter()
                .map(|key| key.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?,
            Slice::new_row_major(
                values
                    .into_iter()
                    .map(|key| key.extract::<usize>())
                    .collect::<PyResult<Vec<usize>>>()?,
            ),
        )
        .map_err(|e| PyValueError::new_err(format!("Invalid shape: {:?}", e)))?;

        Ok(Self {
            inner: AllocSpec {
                shape,
                // TODO(osamas): Support constraints
                constraints: AllocConstraints::none(),
            },
        })
    }
}

#[pyclass(name = "LocalAllocator", module = "monarch._monarch.hyperactor")]
pub struct PyLocalAllocator;

#[pymethods]
impl PyLocalAllocator {
    #[classmethod]
    fn allocate<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        spec: &PyAllocSpec,
    ) -> PyResult<Bound<'py, PyAny>> {
        // We could use Bound here, and acquire the GIL inside of `future_into_py`, but
        // it is rather awkward with the current APIs, and we can anyway support Arc/Mutex
        // pretty easily.
        let spec = spec.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalAllocator
                .allocate(spec)
                .await
                .map(|inner| PyLocalAlloc {
                    inner: Arc::new(std::sync::Mutex::new(Some(inner))),
                })
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })
    }
}

/// Helper trait that allows us to abstract over the different kinds of PyAlloc.
pub trait TakeableAlloc<T> {
    fn take(&self) -> Option<T>;
}

#[pyclass(name = "LocalAlloc", module = "monarch._monarch.hyperactor")]
#[derive(Clone)]
pub struct PyLocalAlloc {
    inner: Arc<std::sync::Mutex<Option<LocalAlloc>>>,
}

impl TakeableAlloc<LocalAlloc> for PyLocalAlloc {
    fn take(&self) -> Option<LocalAlloc> {
        self.inner.lock().unwrap().take()
    }
}

#[pyclass(name = "ProcessAllocator", module = "monarch._monarch.hyperactor")]
pub struct PyProcessAllocator {
    inner: Arc<tokio::sync::Mutex<ProcessAllocator>>,
}

#[pymethods]
impl PyProcessAllocator {
    #[new]
    #[pyo3(signature = (cmd, args=None, env=None))]
    fn new(cmd: String, args: Option<Vec<String>>, env: Option<HashMap<String, String>>) -> Self {
        let mut cmd = Command::new(cmd);
        if let Some(args) = args {
            cmd.args(args);
        }
        if let Some(env) = env {
            cmd.envs(env);
        }
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(ProcessAllocator::new(cmd))),
        }
    }

    fn allocate<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<Bound<'py, PyAny>> {
        // We could use Bound here, and acquire the GIL inside of `future_into_py`, but
        // it is rather awkward with the current APIs, and we can anyway support Arc/Mutex
        // pretty easily.
        let instance = Arc::clone(&self.inner);
        let spec = spec.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            instance
                .lock()
                .await
                .allocate(spec)
                .await
                .map(|inner| PyProcessAlloc {
                    inner: Arc::new(std::sync::Mutex::new(Some(inner))),
                })
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })
    }
}

#[pyclass(name = "ProcessAlloc", module = "monarch._monarch.hyperactor")]
#[derive(Clone)]
pub struct PyProcessAlloc {
    inner: Arc<std::sync::Mutex<Option<ProcessAlloc>>>,
}

impl TakeableAlloc<ProcessAlloc> for PyProcessAlloc {
    fn take(&self) -> Option<ProcessAlloc> {
        self.inner.lock().unwrap().take()
    }
}
