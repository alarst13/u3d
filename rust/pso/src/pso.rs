// PSO Optimization Implementation
// Inspired by the PSO algorithm implementation by Lj Miranda
// Original Python implementation: https://github.com/ljvmiranda921/pyswarms
use ndarray::{Array1, Array2, ArrayView1};
use rand::{distributions::Uniform, Rng};

pub(crate) fn pso<F>(
    func: F,
    lb: ArrayView1<f64>,
    ub: ArrayView1<f64>,
    ieqcons: Option<Box<dyn Fn(ArrayView1<f64>) -> Array1<f64>>>,
    f_ieqcons: Option<Box<dyn Fn(ArrayView1<f64>) -> Array1<f64>>>,
    args: (),
    kwargs: (),
    swarmsize: Option<usize>,
    omega: Option<f64>,
    phip: Option<f64>,
    phig: Option<f64>,
    maxiter: Option<usize>,
    minstep: Option<f64>,
    minfunc: Option<f64>,
    debug: Option<bool>,
) -> (Array1<f64>, f64)
where
    F: Fn(ArrayView1<f64>, (), ()) -> f64,
{
    // Set default values for parameters
    let default_swarmsize = 100;
    let default_omega = 0.5;
    let default_phip = 0.5;
    let default_phig = 0.5;
    let default_maxiter = 100;
    let default_minstep = 1e-8;
    let default_minfunc = 1e-8;
    let default_debug = false;

    let swarmsize = swarmsize.unwrap_or(default_swarmsize);
    let omega = omega.unwrap_or(default_omega);
    let phip = phip.unwrap_or(default_phip);
    let phig = phig.unwrap_or(default_phig);
    let maxiter = maxiter.unwrap_or(default_maxiter);
    let minstep = minstep.unwrap_or(default_minstep);
    let minfunc = minfunc.unwrap_or(default_minfunc);
    let debug = debug.unwrap_or(default_debug);

    assert_eq!(
        lb.len(),
        ub.len(),
        "Lower and upper bounds must be the same size"
    );
    assert!(
        lb.iter().zip(ub.iter()).all(|(l, u)| l < u),
        "All lower bounds must be strictly less than upper bounds"
    );

    let vhigh = &(&ub - &lb).mapv(f64::abs);
    let vlow = -vhigh;

    // Check for constraint function(s) ########################################
    let obj = |x: ArrayView1<f64>| func(x, args, kwargs);
    let cons: Box<dyn Fn(ArrayView1<f64>) -> Array1<f64>> = match f_ieqcons {
        None => {
            match &ieqcons {
                None => {
                    if debug {
                        println!("No constraints given.")
                    }
                    Box::new(|_| Array1::zeros(0))
                }
                Some(ieqcons) => {
                    if debug {
                        println!("Converting ieqcons to a function of the form f_ieqcons(x, args, kwargs).")
                    }
                    Box::new(move |x| ieqcons(x))
                }
            }
        }
        Some(f_ieqcons) => {
            if debug {
                println!("Single constraint function given in f_ieqcons.")
            }
            f_ieqcons
        }
    };

    let is_feasible = |x: ArrayView1<f64>| cons(x).iter().all(|&c| c >= 0.0);

    // Initialize the particle swarm ###########################################
    let s = swarmsize;
    let d = lb.len(); // The number of dimensions each particle has
    let mut rng = rand::thread_rng();

    let mut x = Array2::from_shape_fn((s, d), |(_, _)| rand::thread_rng().gen_range(0.0..1.0));
    let mut v: Array2<f64> = Array2::zeros(x.raw_dim()); // Initialize the particles velocities
    let mut p: Array2<f64> = Array2::zeros(x.raw_dim()); // Initialize the particles' best known positions
    let mut fp: Array1<f64> = Array1::zeros(s); // and the particles' best known positions' values
    let mut g = Array1::zeros(d); // Initialize the swarm's best known position
    let mut fg = f64::INFINITY; // and the best known position's value

    for i in 0..s {
        // Initialize the particle's position
        let x_row = x.row(i).to_owned();
        x.row_mut(i).assign(&(lb.to_owned() + &x_row * (&ub - &lb)));

        // Initialize the particle's best known position
        p.row_mut(i).assign(&x.row(i));

        // Calculate the objective's value at the current particle's
        fp[i] = obj(p.row(i).view());

        // At the start, there may not be any feasible starting point, so just
        // give it a temporary "best" point since it's likely to change
        if i == 0 {
            g.assign(&p.row(0));
        }

        // If the current particle's position is better than the swarm's,
        // update the best swarm position
        if fp[i] < fg && is_feasible(p.row(i).view()) {
            fg = fp[i];
            g.assign(&p.row(i));
        }

        // Initialize the particle's velocity
        v.row_mut(i).assign(
            &(&vlow + Array1::from_shape_fn(d, |_| rng.gen_range(0.0..1.0)) * (vhigh - &vlow)),
        );
    }

    let mut it = 1; // Initialize the iteration counter

    while it <= maxiter {
        let rp_dist = Uniform::new(0.0, 1.0);
        let rg_dist = Uniform::new(0.0, 1.0);
        let rp: Array2<f64> = Array2::from_shape_fn((s, d), |_| rng.sample(rp_dist));
        let rg: Array2<f64> = Array2::from_shape_fn((s, d), |_| rng.sample(rg_dist));

        for i in 0..s {
            let mut v_new = v.row(i).to_owned();
            v_new += &(&v.row(i).mapv(|val| omega * val)
                + &rp.row(i).mapv(|val| phip * val) * (&p.row(i) - &x.row(i))
                + &rg.row(i).mapv(|val| phig * val) * (&g - &x.row(i)));
            v.row_mut(i).assign(&v_new);

            // Update the particle's position
            let mut x_new = x.row(i).to_owned() + &v.row(i);
            x_new.iter_mut().zip(lb.iter()).for_each(|(xi, l)| {
                if *xi < *l {
                    *xi = *l;
                }
            });
            x_new.iter_mut().zip(ub.iter()).for_each(|(xi, u)| {
                if *xi > *u {
                    *xi = *u;
                }
            });
            x.row_mut(i).assign(&x_new);

            // Update the objective function value
            let fx = obj(x.row(i).view());

            // Compare particle's best position (if constraints are satisfied)
            if fx < fp[i] && is_feasible(x.row(i).view()) {
                p.row_mut(i).assign(&x.row(i));
                fp[i] = fx;

                // Compare swarm's best position to current particle's position
                // (Can only get here if constraints are satisfied)
                if fx < fg {
                    if debug {
                        println!(
                            "New best for swarm at iteration {}: {:?} {}",
                            it,
                            x.row(i),
                            fx
                        );
                    }

                    let tmp = x.row(i).to_owned();
                    let stepsize = ((&g - &tmp).mapv(|val| val.powi(2)).sum()).sqrt();
                    if (fg - fx).abs() <= minfunc {
                        println!(
                            "Stopping search: Swarm best objective change less than {}",
                            minfunc
                        );
                        return (tmp, fx);
                    } else if stepsize <= minstep {
                        println!(
                            "Stopping search: Swarm best position change less than {}",
                            minstep
                        );
                        return (tmp, fx);
                    } else {
                        g.assign(&tmp);
                        fg = fx;
                    }
                }
            }
        }

        if debug {
            println!("Best after iteration {}: {:?} {}", it, g, fg);
        }

        it += 1;
    }

    println!(
        "Stopping search: maximum iterations reached --> {}",
        maxiter
    );

    if !is_feasible(g.view()) {
        println!("However, the optimization couldn't find a feasible design. Sorry");
    }

    (g, fg)
}

#[cfg(test)]
mod tests {
    use super::pso;
    use ndarray::{Array1, ArrayView1};

    /* Test function: Sum of squares of elements in the input vector.

    Calculates the sum of squares of elements in the input vector `x`.

    # Arguments

    * `x` - Input array view representing the particle's position.
    * `_` - Placeholder for additional arguments (not used).
    * `_` - Placeholder for additional keyword arguments (not used).

    # Returns

    Returns the sum of squares of elements in the input vector `x`.
    */
    fn test_function(x: ArrayView1<f64>, _: (), _: ()) -> f64 {
        x.iter().map(|&val| val * val).sum()
    }

    /* Test constraints function: Enforces non-negativity constraints.

    This function enforces non-negativity constraints on each element of the input vector `x`.
    It maps the input vector to a new vector where each element is the maximum of the difference
    between the original element and 0.0, ensuring that the resulting vector satisfies the
    non-negativity constraints.

    # Arguments

    * `x` - Input array view representing the particle's position.

    # Returns

    Returns a new array where each element enforces non-negativity constraints on the corresponding
    element of the input vector `x`.
    */
    fn test_constraints(x: ArrayView1<f64>) -> Array1<f64> {
        x.map(|val| val - 0.0).map(|diff| diff.max(0.0)) // Ensure that differences are non-negative
    }

    #[test]
    fn test_pso_optimization() {
        let lb = Array1::from(vec![-5.0, -5.0]);
        let ub = Array1::from(vec![5.0, 5.0]);

        /* Call the PSO optimization function with the defined parameters:
           - test_function: Objective function to minimize, which calculates the sum of squares of input values.
           - lb.view(): Lower bounds of the search space.
           - ub.view(): Upper bounds of the search space.
           - None: No equality constraint function provided.
           - Some(Box::new(test_constraints)): Constraint function to ensure non-negativity of elements.
           - (): Empty tuple for additional arguments to the objective function.
           - (): Empty tuple for additional keyword arguments to the objective function.
           - Some(100): Maximum number of iterations.
           - Some(0.5): Inertia weight parameter.
           - Some(0.5): Personal acceleration coefficient.
           - Some(0.5): Global acceleration coefficient.
           - Some(100): Maximum number of iterations for PSO algorithm.
           - Some(1e-8): Minimum step size for termination.
           - Some(1e-8): Minimum change in objective function value for termination.
           - false: Debug flag.

           The PSO optimization will attempt to minimize the output of the test_function by finding the input
           values that produce the smallest sum of squares, subject to the constraint of non-negativity enforced
           by the test_constraints function.
        */
        let (best_position, best_value) = pso(
            test_function,
            lb.view(),
            ub.view(),
            None,
            Some(Box::new(test_constraints)),
            (),
            (),
            Some(100),
            Some(0.5),
            Some(0.5),
            Some(0.5),
            Some(100),
            Some(1e-8),
            Some(1e-8),
            Some(true),
        );

        // Check if the optimal value is within an acceptable range
        let expected_optimal_value = 0.0; // Expected optimal value for the given test_function
        let tolerance = 0.5; // Tolerance level for accepting the optimal value
        assert!(
            (best_value - expected_optimal_value).abs() < tolerance,
            "Optimal value is not within an acceptable range"
        );

        // Check if the optimal position is within the defined bounds
        let lower_bounds = lb.to_owned();
        let upper_bounds = ub.to_owned();
        assert!(
            best_position
                .iter()
                .zip(lower_bounds.iter())
                .all(|(val, lb)| val >= lb)
                && best_position
                    .iter()
                    .zip(upper_bounds.iter())
                    .all(|(val, ub)| val <= ub),
            "Optimal position is outside the defined bounds"
        );
    }
}
