use ndarray::{Array1, Array2, s};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::f64::consts::E;

#[pyfunction]
fn calc_lle_py(
    py: Python,
    alpha: &PyArray2<f64>,
    tau: &PyArray2<f64>,
    z: &PyArray1<f64>,
    x0: &PyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, f64)> {
    let alpha_array = unsafe { alpha.as_array().to_owned() };
    let tau_array = unsafe { tau.as_array().to_owned() };
    let z_array = unsafe { z.as_array().to_owned() };
    let x0_array = unsafe { x0.as_array().to_owned() };

    let (x, y, beta) = calc_lle(&alpha_array, &tau_array, &z_array, &x0_array);
    Ok((x.into_pyarray(py).to_owned(), y.into_pyarray(py).to_owned(), beta.to_owned()))
}

#[pyfunction]
fn get_gamma_py(
    py: Python,
    x: &PyArray1<f64>,
    alpha: &PyArray2<f64>,
    tau: &PyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_array = unsafe { x.as_array().to_owned() };
    let alpha_array = unsafe { alpha.as_array().to_owned() };
    let tau_array = unsafe { tau.as_array().to_owned() };

    let gamma_x = get_gamma(&x_array, &alpha_array, &tau_array);

    Ok(gamma_x.into_pyarray(py).to_owned())
}


#[pymodule]
fn py_nrtl(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(calc_lle_py))?;
    m.add_wrapped(wrap_pyfunction!(get_gamma_py))?;

    Ok(())
}

fn calc_lle(
    alpha: &Array2<f64>,
    tau: &Array2<f64>,
    z: &Array1<f64>,
    x0: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, f64) {
    let beta = 0.5;
    let n_comp = z.len();
    let nitermax = 200;
    let tol_mu = 1e-10;
    let tol_beta = 1e-10;
    let tol_gbeta = 1e-10;

    let mut beta_out = 0.0;
    let mut x = x0.clone();
    let mut y = Array1::<f64>::zeros(n_comp);

    for i in 0..(n_comp - 1) {
        y[i] = (z[i] - (1.0 - beta) * x[i]) / beta;
    }

    y[n_comp - 1] = 1.0 - y.slice(s![..n_comp - 1]).sum();

    let mut gamma_x = get_gamma(&x, alpha, tau);
    let mut gamma_y = get_gamma(&y, alpha, tau);
    let mut k = &gamma_x / &gamma_y;
    let mut delta_mu = (&gamma_y * &y - &gamma_x * &x).mapv(f64::abs);

    let mut nit = 0;
    while nit < nitermax && delta_mu.iter().cloned().fold(f64::MIN, f64::max) > tol_mu {
        nit += 1;

        let (beta_new, x_new, y_new) = rrsolver(n_comp, z, &k, nitermax, tol_beta, tol_gbeta);

        if beta_new < tol_beta || beta_new > (1.0 - tol_beta)  {
            x = z.clone();
            y = z.clone();
            break;
        }

        x = x_new;
        y = y_new;
        gamma_x = get_gamma(&x, alpha, tau);
        gamma_y = get_gamma(&y, alpha, tau);

        k = &gamma_x / &gamma_y;
        delta_mu = (&gamma_y * &y - &gamma_x * &x).mapv(f64::abs);
        beta_out = beta_new;
    }

    (x, y, beta_out)
}

#[inline]
fn get_gamma(x: &Array1<f64>, alpha: &Array2<f64>, tau: &Array2<f64>) -> Array1<f64> {
    let n_comp = x.len();

    // G
    let mut g = Array2::<f64>::eye(n_comp);
    for i in 0..n_comp {
        for j in 0..n_comp {
            if i != j {
                g[[i, j]] = E.powf(-alpha[[i, j]] * tau[[i, j]]);
            }
        }
    }

    // B(i)
    let mut b = Array1::<f64>::zeros(n_comp);
    for i in 0..n_comp {
        b[i] = (0..n_comp).map(|j| tau[[j, i]] * g[[j, i]] * x[j]).sum();
    }

    // A(i)
    let mut a = Array1::<f64>::zeros(n_comp);
    for i in 0..n_comp {
        a[i] = (0..n_comp).map(|l| g[[l, i]] * x[l]).sum();
    }

    // gamma(i)
    let mut summe_lngamma = Array1::<f64>::zeros(n_comp);
    for i in 0..n_comp {
        summe_lngamma[i] = (0..n_comp)
            .map(|j| x[j] * g[[i, j]] / a[j] * (tau[[i, j]] - b[j] / a[j]))
            .sum();
    }

    let mut gamma = Array1::<f64>::zeros(n_comp);
    for i in 0..n_comp {
        let lngamma = b[i] / a[i] + summe_lngamma[i];
        gamma[i] = lngamma.exp();
    }

    gamma
}

#[inline]
fn rrsolver(
    nc: usize,
    z: &Array1<f64>,
    k: &Array1<f64>,
    ni: usize,
    tol_beta: f64,
    tol_gbeta: f64,
) -> (f64, Array1<f64>, Array1<f64>) {
    let mut beta = 0.5;
    let mut beta_min = 1e-8;
    let mut beta_max = 1.0 - beta_min;

    let mut delta_beta = 1.0;

    for _ in 0..ni {
        let mut g = 0.0;
        let g_alt = g;
        for i in 0..nc {
            g += z[i] * (k[i] - 1.0) / (1.0 - beta + beta * k[i]);
        }

        if g < 0.0 {
            beta_max = beta;
        } else {
            beta_min = beta;
        }

        let mut g_strich = 0.0;

        for i in 0..nc {
            g_strich -= (z[i].sqrt() * (k[i] - 1.0) / (beta * k[i] - beta + 1.0)).powi(2);
        }

        let beta_neu = beta - g / g_strich;

        if beta_neu >= beta_min && beta_neu <= beta_max && ni > 1 {
            delta_beta = (beta - beta_neu).abs();
            beta = beta_neu;
        } else {
            let beta_neu = (beta_min + beta_max) / 2.0;
            beta = beta_neu;
        }

        let delta_g = (g_alt - g).abs();

        if delta_beta <= tol_beta && delta_g <= tol_gbeta {
            break;
        }
    }

    let mut l = Array1::<f64>::zeros(nc);
    let mut v = Array1::<f64>::zeros(nc);
    let mut x = Array1::<f64>::zeros(nc);
    let mut y = Array1::<f64>::zeros(nc);

    if ni > 1 {
        for i in 0..nc {
            l[i] = (1.0 - beta) * z[i] / (1.0 - beta + beta * k[i]);
            v[i] = (beta * k[i] * z[i]) / (1.0 - beta + beta * k[i]);
            x[i] = l[i] / (1.0 - beta);
            y[i] = v[i] / beta;
        }
    }

    (beta, x, y)
}