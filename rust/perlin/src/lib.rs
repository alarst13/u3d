use pyo3::prelude::*;

#[macro_use]
extern crate lazy_static;

mod perlin3d;

use perlin3d::perlin_with_octaves;

#[pyfunction]
fn perlin_noise(
    x: f64,
    y: f64,
    t: f64,
    num_octaves: usize,
    wavelength_x: f64,
    wavelength_y: f64,
    wavelength_z: f64,
    color_period: f64,
) -> PyResult<f64> {
    Ok(perlin_with_octaves(
        x,
        y,
        t,
        num_octaves,
        wavelength_x,
        wavelength_y,
        wavelength_z,
        color_period,
    ))
}

/*
A Python module implemented in Rust. The name of this function must match
the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
import the module.
*/
#[pymodule]
fn perlin(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perlin_noise, m)?)?;
    Ok(())
}
