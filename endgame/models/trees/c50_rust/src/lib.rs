//! C5.0 Decision Tree Implementation
//!
//! A high-performance, multi-threaded implementation of the C5.0 decision tree algorithm
//! with Python bindings via PyO3.

mod types;
mod info;
mod split;
mod tree;
mod prune;
mod classify;
mod boost;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array2;

pub use types::*;
pub use tree::C50Tree;
pub use boost::C50Ensemble;

/// Python module for C5.0 decision tree
#[pymodule]
fn c50_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyC50Classifier>()?;
    m.add_class::<PyC50Ensemble>()?;
    Ok(())
}

/// Python wrapper for C5.0 classifier
#[pyclass(name = "C50Classifier")]
pub struct PyC50Classifier {
    tree: Option<C50Tree>,
    n_classes: usize,
    n_features: usize,
    config: TreeConfig,
}

#[pymethods]
impl PyC50Classifier {
    #[new]
    #[pyo3(signature = (
        min_cases = 2,
        cf = 0.25,
        use_subset = true,
        global_pruning = true,
        soft_threshold = false,
        n_threads = 0,
        random_state = None
    ))]
    fn new(
        min_cases: usize,
        cf: f64,
        use_subset: bool,
        global_pruning: bool,
        soft_threshold: bool,
        n_threads: usize,
        random_state: Option<u64>,
    ) -> Self {
        PyC50Classifier {
            tree: None,
            n_classes: 0,
            n_features: 0,
            config: TreeConfig {
                min_cases,
                cf,
                use_subset,
                global_pruning,
                soft_threshold,
                n_threads: if n_threads == 0 {
                    rayon::current_num_threads()
                } else {
                    n_threads
                },
                random_state,
            },
        }
    }

    /// Fit the classifier to training data
    #[pyo3(signature = (x, y, feature_names = None, categorical_features = None, sample_weight = None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i64>,
        feature_names: Option<Vec<String>>,
        categorical_features: Option<Vec<usize>>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        if y_array.len() != n_samples {
            return Err(PyValueError::new_err("X and y must have same number of samples"));
        }

        self.n_features = n_features;

        // Convert to internal format
        let mut dataset = Dataset::new(n_samples, n_features);

        for i in 0..n_samples {
            for j in 0..n_features {
                dataset.set_value(i, j, x_array[[i, j]]);
            }
            dataset.set_class(i, y_array[i] as usize);
        }

        // Set weights if provided
        if let Some(weights) = sample_weight {
            let w = weights.as_array();
            for i in 0..n_samples {
                dataset.set_weight(i, w[i]);
            }
        }

        // Mark categorical features
        if let Some(cats) = categorical_features {
            for idx in cats {
                if idx < n_features {
                    dataset.set_categorical(idx, true);
                }
            }
        }

        // Determine number of classes
        self.n_classes = dataset.n_classes();

        // Build tree
        py.allow_threads(|| {
            let mut builder = tree::TreeBuilder::new(&dataset, self.config.clone());
            self.tree = Some(builder.build());
        });

        Ok(())
    }

    /// Predict class labels
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<&'py PyArray1<i64>> {
        let tree = self.tree.as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;

        let x_array = x.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let predictions: Vec<i64> = py.allow_threads(|| {
            (0..n_samples)
                .map(|i| {
                    let row: Vec<f64> = (0..n_features)
                        .map(|j| x_array[[i, j]])
                        .collect();
                    tree.classify(&row) as i64
                })
                .collect()
        });

        Ok(PyArray1::from_vec(py, predictions))
    }

    /// Predict class probabilities
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let tree = self.tree.as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;

        let x_array = x.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();
        let n_classes = self.n_classes;

        let probas: Vec<f64> = py.allow_threads(|| {
            let mut result = Vec::with_capacity(n_samples * n_classes);
            for i in 0..n_samples {
                let row: Vec<f64> = (0..n_features)
                    .map(|j| x_array[[i, j]])
                    .collect();
                let class_probs = tree.classify_proba(&row);
                for c in 0..n_classes {
                    result.push(if c < class_probs.len() { class_probs[c] } else { 0.0 });
                }
            }
            result
        });

        // Convert to 2D array
        let arr = Array2::from_shape_vec((n_samples, n_classes), probas)
            .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;

        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Get feature importances
    fn feature_importances<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        let tree = self.tree.as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;

        let importances = tree.feature_importances();
        Ok(PyArray1::from_vec(py, importances))
    }

    /// Get tree depth
    fn get_depth(&self) -> PyResult<usize> {
        let tree = self.tree.as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;
        Ok(tree.depth())
    }

    /// Get number of leaves
    fn get_n_leaves(&self) -> PyResult<usize> {
        let tree = self.tree.as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;
        Ok(tree.n_leaves())
    }
}

/// Python wrapper for boosted C5.0 ensemble
#[pyclass(name = "C50Ensemble")]
pub struct PyC50Ensemble {
    ensemble: Option<C50Ensemble>,
    n_classes: usize,
    n_features: usize,
    config: TreeConfig,
    n_trials: usize,
}

#[pymethods]
impl PyC50Ensemble {
    #[new]
    #[pyo3(signature = (
        n_trials = 10,
        min_cases = 2,
        cf = 0.25,
        use_subset = true,
        n_threads = 0,
        random_state = None
    ))]
    fn new(
        n_trials: usize,
        min_cases: usize,
        cf: f64,
        use_subset: bool,
        n_threads: usize,
        random_state: Option<u64>,
    ) -> Self {
        PyC50Ensemble {
            ensemble: None,
            n_classes: 0,
            n_features: 0,
            config: TreeConfig {
                min_cases,
                cf,
                use_subset,
                global_pruning: true,
                soft_threshold: false,
                n_threads: if n_threads == 0 {
                    rayon::current_num_threads()
                } else {
                    n_threads
                },
                random_state,
            },
            n_trials,
        }
    }

    #[pyo3(signature = (x, y, categorical_features = None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i64>,
        categorical_features: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        self.n_features = n_features;

        let mut dataset = Dataset::new(n_samples, n_features);

        for i in 0..n_samples {
            for j in 0..n_features {
                dataset.set_value(i, j, x_array[[i, j]]);
            }
            dataset.set_class(i, y_array[i] as usize);
        }

        if let Some(cats) = categorical_features {
            for idx in cats {
                if idx < n_features {
                    dataset.set_categorical(idx, true);
                }
            }
        }

        self.n_classes = dataset.n_classes();

        py.allow_threads(|| {
            let mut booster = boost::Booster::new(&dataset, self.config.clone(), self.n_trials);
            self.ensemble = Some(booster.build());
        });

        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<&'py PyArray1<i64>> {
        let ensemble = self.ensemble.as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;

        let x_array = x.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let predictions: Vec<i64> = py.allow_threads(|| {
            (0..n_samples)
                .map(|i| {
                    let row: Vec<f64> = (0..n_features)
                        .map(|j| x_array[[i, j]])
                        .collect();
                    ensemble.classify(&row) as i64
                })
                .collect()
        });

        Ok(PyArray1::from_vec(py, predictions))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let ensemble = self.ensemble.as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;

        let x_array = x.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();
        let n_classes = self.n_classes;

        let probas: Vec<f64> = py.allow_threads(|| {
            let mut result = Vec::with_capacity(n_samples * n_classes);
            for i in 0..n_samples {
                let row: Vec<f64> = (0..n_features)
                    .map(|j| x_array[[i, j]])
                    .collect();
                let class_probs = ensemble.classify_proba(&row);
                for c in 0..n_classes {
                    result.push(if c < class_probs.len() { class_probs[c] } else { 0.0 });
                }
            }
            result
        });

        let arr = Array2::from_shape_vec((n_samples, n_classes), probas)
            .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;

        Ok(PyArray2::from_owned_array(py, arr))
    }
}
