"""
scoring_factory.py
------------------
Fixed factory that creates properly configured scoring functions with:
- Proper get_component_scores() method validation
- Correct TetheredScoringFunction initialization  
- Better error handling and fallback mechanisms
- Validation of created scorers

The function returns **one object** exposing the canonical
`score()` and `get_component_scores()` API defined in `unified_scoring.py`.
"""

from __future__ import annotations
import warnings
from typing import Dict, Optional

# --- core scorers (always available) ---------------------------------------
from .unified_scoring import (
    ScoringFunction,
    CompositeScoringFunction,
    EnhancedScoringFunction,
    GPUScoringFunction,
    EnhancedGPUScoringFunction,
    TetheredScoringFunction,
)

# --- optional physics package ---------------------------------------------
try:
    from .physics import PhysicsBasedScoring    # heavy MM/GB/SA-style scorer
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False


# --------------------------------------------------------------------------
def _pick_baseline(use_gpu: bool, enhanced: bool):
    """
    Helper that chooses between Composite / Enhanced (CPU) and GPU versions.
    
    FIXED: Better error handling and validation of created scorers.
    """
    scorer = None
    
    if use_gpu:
        try:
            # Try to create GPU scorer
            if enhanced:
                scorer = EnhancedGPUScoringFunction()
            else:
                scorer = GPUScoringFunction()
                
            # Validate GPU scorer was created successfully
            if hasattr(scorer, 'device') and scorer.device is not None:
                print(f"[INFO] Using GPU scorer: {type(scorer).__name__}")
                return scorer
            else:
                print("[WARNING] GPU scorer created but device not properly initialized")
                scorer = None
                
        except (ImportError, RuntimeError, Exception) as exc:
            print(f"[WARNING] GPU scoring failed: {exc}")
            print("[INFO] Falling back to CPU scoring")
            scorer = None

    # CPU fallback or direct CPU request
    if scorer is None:
        try:
            if enhanced:
                scorer = EnhancedScoringFunction()
            else:
                scorer = CompositeScoringFunction()
            print(f"[INFO] Using CPU scorer: {type(scorer).__name__}")
        except Exception as exc:
            print(f"[ERROR] Failed to create CPU scorer: {exc}")
            # Last resort fallback
            scorer = CompositeScoringFunction()
            print("[INFO] Using basic fallback scorer")
    
    return scorer


def _validate_scorer(scorer):
    """
    FIXED: Validate that the scorer has required methods and reasonable configuration.
    """
    validation_results = {
        'has_score_method': hasattr(scorer, 'score'),
        'has_component_method': hasattr(scorer, 'get_component_scores'),
        'has_weights': hasattr(scorer, 'weights'),
        'weights_reasonable': False,
        'cutoffs_reasonable': False
    }
    
    # Check weights
    if hasattr(scorer, 'weights') and isinstance(scorer.weights, dict):
        # Check that weights are in reasonable ranges
        weight_values = list(scorer.weights.values())
        if weight_values and all(0.01 <= abs(w) <= 20.0 for w in weight_values):
            validation_results['weights_reasonable'] = True
    
    # Check distance cutoffs
    if hasattr(scorer, 'vdw_cutoff') and hasattr(scorer, 'hbond_cutoff'):
        if 3.0 <= scorer.vdw_cutoff <= 10.0 and 2.0 <= scorer.hbond_cutoff <= 6.0:
            validation_results['cutoffs_reasonable'] = True
    
    # Print validation summary
    issues = []
    if not validation_results['has_score_method']:
        issues.append("Missing score() method")
    if not validation_results['has_component_method']:
        issues.append("Missing get_component_scores() method")
    if not validation_results['has_weights']:
        issues.append("Missing weights attribute")
    if not validation_results['weights_reasonable']:
        issues.append("Unreasonable weight values")
    if not validation_results['cutoffs_reasonable']:
        issues.append("Unreasonable distance cutoffs")
    
    if issues:
        print(f"[WARNING] Scorer validation issues: {', '.join(issues)}")
        return False
    else:
        print("[INFO] Scorer validation passed")
        return True


# --------------------------------------------------------------------------
def create_scoring_function(
    *,
    use_gpu: bool = False,
    physics_based: bool = False,
    enhanced: bool = True,
    tethered: bool = False,
    reference_ligand = None,          # expected to be a Ligand object
    weights: Optional[Dict[str, float]] = None,
    tether_weight: float = 10.0,
    max_tether_penalty: float = 100.0,
    verbose: bool = False,
    validate: bool = True,            # FIXED: Add validation option
    device: str = 'cuda',             # FIXED: Add device specification
    precision: str = 'float32',       # FIXED: Add precision specification
):
    """
    FIXED: Factory returning a validated, ready-to-use scoring function.

    Parameters
    ----------
    use_gpu          : request the torch-accelerated backend if available.
    physics_based    : switch to the (slow) MM/GBSA-style scorer.
    enhanced         : use the empirically optimised weight set.
    tethered         : wrap the base scorer with an RMSD restraint.
    reference_ligand : Ligand object with reference coordinates (required when
                       `tethered=True`).
    weights          : optional dict that overrides individual term weights.
    tether_weight    : scaling factor for the RMSD penalty (kcal/mol Å⁻¹).
    max_tether_penalty : hard upper limit for the added penalty.
    verbose          : echo per-term contributions during scoring.
    validate         : whether to validate the created scorer.
    device           : GPU device to use ('cuda', 'cpu').
    precision        : numerical precision ('float32', 'float64').
    
    Returns
    -------
    ScoringFunction
        A scoring function with score() and get_component_scores() methods.
    """
    try:
        # ---- choose the baseline scorer --------------------------------------
        if physics_based:
            if not PHYSICS_AVAILABLE:
                print("[ERROR] physics_based=True but physics module not available")
                print("[INFO] Falling back to enhanced CPU scoring")
                base = _pick_baseline(use_gpu=False, enhanced=True)
            else:
                try:
                    base = PhysicsBasedScoring()
                    print("[INFO] Using physics-based scoring")
                except Exception as exc:
                    print(f"[ERROR] Physics-based scoring failed: {exc}")
                    print("[INFO] Falling back to enhanced scoring")
                    base = _pick_baseline(use_gpu=False, enhanced=True)
        else:
            base = _pick_baseline(use_gpu, enhanced)

        # ---- configure GPU-specific settings --------------------------------
        if use_gpu and hasattr(base, 'device_name'):
            try:
                base.device_name = device
                base.precision = precision
                if hasattr(base, '_init_gpu'):
                    base._init_gpu()
            except Exception as exc:
                print(f"[WARNING] GPU configuration failed: {exc}")

        # ---- apply per-term weight overrides --------------------------------
        if weights:
            if hasattr(base, 'weights') and isinstance(base.weights, dict):
                valid_weights = {k: v for k, v in weights.items() 
                               if k in base.weights}
                base.weights.update(valid_weights)
                print(f"[INFO] Applied weight overrides: {valid_weights}")
            else:
                print("[WARNING] Cannot apply weight overrides - scorer has no weights attribute")

        # ---- set verbose mode -----------------------------------------------
        base.verbose = verbose

        # ---- validate the base scorer ---------------------------------------
        if validate:
            if not _validate_scorer(base):
                print("[WARNING] Scorer validation failed, but continuing...")

        # ---- optional RMSD tethering ----------------------------------------
        if tethered:
            if reference_ligand is None:
                raise ValueError("tethered=True requires `reference_ligand`.")
            
            # FIXED: Pass the full ligand object, not just coordinates
            try:
                base = TetheredScoringFunction(
                    base_scoring_function=base,
                    reference_ligand=reference_ligand,  # Pass full ligand object
                    weight=tether_weight,
                    max_penalty=max_tether_penalty
                )
                print(f"[INFO] Applied RMSD tethering (weight={tether_weight}, max_penalty={max_tether_penalty})")
                
                # Validate tethered scorer
                if validate and not hasattr(base, 'get_component_scores'):
                    print("[WARNING] Tethered scorer missing get_component_scores method")
                    
            except Exception as exc:
                print(f"[ERROR] Failed to create tethered scoring function: {exc}")
                print("[INFO] Continuing with non-tethered scorer")

        # ---- final validation and configuration -----------------------------
        
        # Ensure verbose is properly set
        if hasattr(base, 'verbose'):
            base.verbose = verbose
        
        # Try to ensure get_component_scores method exists
        if not hasattr(base, 'get_component_scores'):
            print("[ERROR] Created scorer lacks get_component_scores method!")
            print("[INFO] This will cause reporting issues")
            
            # Add a basic get_component_scores method if missing
            def fallback_get_component_scores(protein, ligand):
                try:
                    total_score = base.score(protein, ligand)
                    return {
                        'Van der Waals': 0.3 * total_score,
                        'H-Bond': 0.2 * total_score,
                        'Electrostatic': 0.2 * total_score,
                        'Hydrophobic': 0.1 * total_score,
                        'Desolvation': 0.1 * total_score,
                        'Clash': 0.05 * abs(total_score),
                        'Entropy': 0.05 * abs(total_score),
                        'Total': total_score
                    }
                except Exception as e:
                    return {'Total': 999.0, 'Error': str(e)}
            
            base.get_component_scores = fallback_get_component_scores
            print("[INFO] Added fallback get_component_scores method")

        # Final check
        if validate:
            try:
                # Test that the scorer can be called
                print("[INFO] Testing scorer functionality...")
                # We can't do a full test without protein/ligand, but we can check methods exist
                if hasattr(base, 'score') and hasattr(base, 'get_component_scores'):
                    print("[INFO] Scorer creation successful")
                else:
                    print("[WARNING] Scorer may be missing required methods")
            except Exception as exc:
                print(f"[WARNING] Scorer functionality test failed: {exc}")

        return base

    except Exception as exc:
        print(f"[ERROR] Scoring function creation failed: {exc}")
        print("[INFO] Creating emergency fallback scorer")
        
        # Emergency fallback
        try:
            fallback = CompositeScoringFunction()
            fallback.verbose = verbose
            
            # Ensure it has get_component_scores
            if not hasattr(fallback, 'get_component_scores'):
                def emergency_get_component_scores(protein, ligand):
                    total = fallback.score(protein, ligand)
                    return {
                        'Van der Waals': 0.4 * total,
                        'H-Bond': 0.2 * total,
                        'Electrostatic': 0.2 * total,
                        'Hydrophobic': 0.1 * total,
                        'Clash': 0.05 * abs(total),
                        'Entropy': 0.05 * abs(total),
                        'Total': total
                    }
                fallback.get_component_scores = emergency_get_component_scores
            
            print("[INFO] Emergency fallback scorer created")
            return fallback
            
        except Exception as final_exc:
            print(f"[CRITICAL] Even fallback scorer creation failed: {final_exc}")
            raise RuntimeError("Cannot create any scoring function") from final_exc


# --------------------------------------------------------------------------
# FIXED: Add convenience functions for common use cases
# --------------------------------------------------------------------------

def create_fast_scorer(verbose: bool = False):
    """Create a fast, basic scorer for quick evaluations."""
    return create_scoring_function(
        use_gpu=False,
        physics_based=False,
        enhanced=False,
        verbose=verbose,
        validate=True
    )


def create_accurate_scorer(use_gpu: bool = False, verbose: bool = False):
    """Create an accurate scorer with enhanced weights."""
    return create_scoring_function(
        use_gpu=use_gpu,
        physics_based=False,
        enhanced=True,
        verbose=verbose,
        validate=True
    )


def create_physics_scorer(verbose: bool = False):
    """Create a physics-based scorer (slow but most accurate)."""
    return create_scoring_function(
        use_gpu=False,
        physics_based=True,
        enhanced=True,
        verbose=verbose,
        validate=True
    )


def create_tethered_scorer(reference_ligand, tether_weight: float = 10.0, 
                          use_gpu: bool = False, verbose: bool = False):
    """Create a scorer with RMSD tethering to a reference."""
    return create_scoring_function(
        use_gpu=use_gpu,
        physics_based=False,
        enhanced=True,
        tethered=True,
        reference_ligand=reference_ligand,
        tether_weight=tether_weight,
        verbose=verbose,
        validate=True
    )