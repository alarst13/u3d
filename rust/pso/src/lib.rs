mod pso;

use numpy::pyo3::IntoPy;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyDict};
use pyo3::wrap_pyfunction;
use ndarray::ArrayView1;
use numpy::{PyArray1, ToPyArray};

use crate::pso::pso;

#[pyfunction]
fn particle_swarm_optimization(
    py: Python<'_>,
    func: PyObject,
    lb: &PyArray1<f64>,
    ub: &PyArray1<f64>,
    ieqcons: Option<PyObject>,
    f_ieqcons: Option<PyObject>,
    args: Option<&PyTuple>,
    kwargs: Option<&PyDict>,
    swarmsize: Option<usize>,
    omega: Option<f64>,
    phip: Option<f64>,
    phig: Option<f64>,
    maxiter: Option<usize>,
    minstep: Option<f64>,
    minfunc: Option<f64>,
    debug: Option<bool>,
) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    let wrapped_func = |x: ArrayView1<f64>, _: (), _: ()| -> f64 {
        Python::with_gil(|pyo3_py| {
            let py_arr = x.to_pyarray(pyo3_py);
            let py_obj: PyObject = py_arr.into_py(pyo3_py);
            match func.call1(pyo3_py, (py_obj,)) {
                Ok(result) => result.extract::<f64>(pyo3_py).unwrap_or(f64::MAX), // use default value if extract fails
                Err(_) => f64::MAX, // or some other value to indicate an error
            }
        })
    };

    let (best_position, best_value) = pso(
        wrapped_func,
        unsafe { lb.as_array().view() },
        unsafe { ub.as_array().view() },
        None, // ieqcons
        None, // f_ieqcons
        (),   // args
        (),   // kwargs
        swarmsize,
        omega,
        phip,
        phig,
        maxiter,
        minstep,
        minfunc,
        debug,
    );

    Ok((best_position.to_pyarray(py).to_owned(), best_value))
}

#[pymodule]
fn psolib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(particle_swarm_optimization, m)?)?;
    Ok(())
}
