"""Surrogate model pipeline entrypoint and orchestrator.

This module provides a command-line interface and a high-level
``PipelineOrchestrator`` class to run the end-to-end workflow for training
and evaluating surrogate models from precomputed inputs (e.g., random forest
probabilities). It coordinates cross-validation, metric reporting, and
optional visualizations such as UMAP projections and decision tree plots.

Typical usage from the CLI:

    python -m _surrogate_model.main --data path/to/predictions.csv --folds 5

Programmatic usage:

    orchestrator = PipelineOrchestrator(data_path, n_splits=5)
    results = orchestrator.run(cluster_mode=True, visualize=True)

"""

import argparse
import pandas as pd
import numpy as np
from modules import (
    LoadData, ClusterData, SurrogateModel,
    EvaluateTraining, ModelReport,
    UMAPProjection, InterpretPredictions,
    CrossValidation
)
from modules.config import (
    DATA_PATH, N_SPLITS, SEED,
    EXCLUDE_COLS, CLUSTER_THRESHOLD
)

class PipelineOrchestrator:
    """High-level orchestrator for the surrogate modeling pipeline.

    The orchestrator coordinates cross-validation, evaluation, reporting, and
    optional visualization steps. It expects a CSV file containing features
    and precomputed model outputs (e.g., a column like ``prob_class_1``) and
    uses configuration constants from ``modules.config`` to determine defaults
    such as data paths and excluded columns.

    Attributes:
        data_path (str): Filesystem path to the input CSV used for training
            and evaluation.
        n_splits (int): Number of cross-validation folds.
        cv_results (list[dict] | None): Aggregated cross-validation results
            produced by ``CrossValidation.run_cv``. Populated after ``run``.
    """
    
    def __init__(self, data_path=DATA_PATH, n_splits=N_SPLITS):
        self.data_path = data_path
        self.n_splits = n_splits
        self.cv_results = None
        
    def run(self, cluster_mode=False, visualize=True):
        """Execute the complete training and evaluation pipeline.

        Orchestrates cross-validation using the provided number of folds,
        persists evaluation metrics, renders performance plots, and optionally
        generates additional visualizations (UMAP projections and a decision
        tree diagram for the final model).

        Args:
            cluster_mode (bool): If True, train separate surrogate models per
                cluster (as determined by upstream logic in the CV step).
            visualize (bool): If True, create UMAP and tree visualizations.

        Returns:
            list[dict]: Cross-validation results emitted by
            ``CrossValidation.run_cv``. The exact schema is determined by the
            implementation of the modules referenced by this orchestrator but
            typically includes metrics such as ``r2`` and ``mae`` per cluster.
        """

        print("="*60)
        print("Starting Surrogate Model Pipeline")
        print("="*60)
        
        # Run cross-validation
        cv = CrossValidation(self.n_splits, self.data_path)
        self.cv_results = cv.run_cv(cluster_mode=cluster_mode)
        
        # Save and visualize results
        EvaluateTraining.save_metrics(self.cv_results)
        
        reporter = ModelReport()
        reporter.plot_performance(self.cv_results)
        
        if visualize:
            self._generate_visualizations(cv)
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        
        return self.cv_results
    
    def _generate_visualizations(self, cv):
        """Generate UMAP and decision tree visualizations.

        Reads the full dataset to compute feature matrices and derives simple
        cluster labels from the ``prob_class_1`` column for coloring the UMAP
        projection. If a final fitted model is present on the passed
        ``cv`` object, also renders a tree visualization using the available
        feature names.

        Args:
            cv: Cross-validation controller instance used by ``run``. It is
                expected to expose a ``final_model`` attribute whose ``model``
                can be visualized when present.
        """
        df = pd.read_csv(self.data_path)
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
        X_full = df[feature_cols].values
        rf_probs_full = df['prob_class_1'].values
        
        # Create binary cluster labels using the configured threshold
        # 1 => prob >= threshold, 0 => prob < threshold
        cluster_labels = (rf_probs_full >= CLUSTER_THRESHOLD).astype(int)
        
        # UMAP projection
        umap_proj = UMAPProjection()
        umap_proj.generate_umap(X_full)
        umap_proj.visualize_umap(cluster_labels)
        
        # Tree visualization
        # If per-cluster final models exist, visualize each; otherwise fallback
        # to a single final model when available.
        if getattr(cv, 'final_models', None):
            for cluster_name, sm in cv.final_models.items():
                if getattr(sm, 'model', None) is not None:
                    InterpretPredictions.visualize_tree(
                        sm.model,
                        feature_cols,
                        filename=f"decision_tree_{cluster_name}.dot"
                    )
        elif getattr(cv, 'final_model', None) is not None and getattr(cv.final_model, 'model', None) is not None:
            InterpretPredictions.visualize_tree(
                cv.final_model.model,
                feature_cols
            )


def main():
    """Command-line entry point for running the pipeline.

    Parses arguments for input data path, clustering behavior, visualization
    toggle, and the number of cross-validation folds, then executes the
    pipeline accordingly and prints a concise summary of final results.

    CLI Arguments:
        --data (str): Path to RF predictions CSV. Defaults to
            ``modules.config.DATA_PATH``.
        --cluster (flag): Train separate models per cluster.
        --no-viz (flag): Skip visualization generation (UMAP and tree).
        --folds (int): Number of cross-validation folds. Defaults to
            ``modules.config.N_SPLITS``.
    """
    parser = argparse.ArgumentParser(
        description="Surrogate Model Training Pipeline"
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default=DATA_PATH,
        help='Path to RF predictions CSV'
    )
    parser.add_argument(
        '--cluster', 
        action='store_true',
        help='Train separate models per cluster'
    )
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--folds', 
        type=int, 
        default=N_SPLITS,
        help='Number of cross-validation folds'
    )
    
    args = parser.parse_args()
    
    pipeline = PipelineOrchestrator(
        data_path=args.data,
        n_splits=args.folds
    )
    
    results = pipeline.run(
        cluster_mode=args.cluster,
        visualize=not args.no_viz
    )
    
    print(f"\nFinal Results Summary:")
    df_results = pd.DataFrame(results)
    print(df_results.groupby('cluster')[['r2', 'mae']].mean())

if __name__ == "__main__":
    main()
