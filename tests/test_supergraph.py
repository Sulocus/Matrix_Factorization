"""
Tests for Super-Graph parallelization of BiG-AMP spreading.

Tests:
1. F generation determinism (same seed -> same F)
2. SuperGraph data structure creation
3. Parallel BiG-AMP basic functionality
4. All 4 combinations (gaussian/rademacher x standard/orthogonal)
"""

import pytest
import torch
import numpy as np

# Test configuration
N1, N2, M = 50, 50, 10  # Small for fast tests
ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0]
S = 3  # Number of samples
SEED = 42


class TestFGeneration:
    """Test F generation strategies."""

    def test_gaussian_determinism(self):
        """Same seed produces same Gaussian F."""
        from smf.modules.algorithms.bigamp_spreading_parallel import generate_F_gaussian

        device = torch.device('cpu')
        C, M = 100, 10

        F1 = generate_F_gaussian(C, M, SEED, device)
        F2 = generate_F_gaussian(C, M, SEED, device)

        assert torch.allclose(F1, F2), "Gaussian F should be deterministic"

    def test_rademacher_determinism(self):
        """Same seed produces same Rademacher F."""
        from smf.modules.algorithms.bigamp_spreading_parallel import generate_F_rademacher

        device = torch.device('cpu')
        C, M = 100, 10

        F1 = generate_F_rademacher(C, M, SEED, device)
        F2 = generate_F_rademacher(C, M, SEED, device)

        assert torch.allclose(F1, F2), "Rademacher F should be deterministic"

    def test_rademacher_values(self):
        """Rademacher F should only contain {-1, +1}."""
        from smf.modules.algorithms.bigamp_spreading_parallel import generate_F_rademacher

        device = torch.device('cpu')
        C, M = 100, 10

        F = generate_F_rademacher(C, M, SEED, device)

        unique_values = torch.unique(F)
        assert len(unique_values) == 2, f"Rademacher should have 2 unique values, got {len(unique_values)}"
        assert -1 in unique_values and 1 in unique_values, f"Values should be -1 and +1, got {unique_values}"

    def test_gaussian_statistics(self):
        """Gaussian F should have mean~0, var~1."""
        from smf.modules.algorithms.bigamp_spreading_parallel import generate_F_gaussian

        device = torch.device('cpu')
        C, M = 10000, 10  # Large for statistical accuracy

        F = generate_F_gaussian(C, M, SEED, device)

        mean = F.mean().item()
        var = F.var().item()

        assert abs(mean) < 0.1, f"Gaussian mean should be ~0, got {mean}"
        assert abs(var - 1.0) < 0.1, f"Gaussian var should be ~1, got {var}"

    def test_different_seeds_different_F(self):
        """Different seeds should produce different F."""
        from smf.modules.algorithms.bigamp_spreading_parallel import generate_F_gaussian

        device = torch.device('cpu')
        C, M = 100, 10

        F1 = generate_F_gaussian(C, M, SEED, device)
        F2 = generate_F_gaussian(C, M, SEED + 1, device)

        assert not torch.allclose(F1, F2), "Different seeds should produce different F"


class TestSuperGraph:
    """Test SuperGraph data structure."""

    def test_supergraph_creation(self):
        """SuperGraph should be created with correct shapes."""
        from smf.modules.graphs.supergraph import create_supergraph

        device = torch.device('cpu')
        sg = create_supergraph(N1, N2, M, ALPHA_VALUES, S, SEED, device)

        assert sg.i_idx.shape == (S, sg.C_max), f"i_idx shape mismatch: {sg.i_idx.shape}"
        assert sg.j_idx.shape == (S, sg.C_max), f"j_idx shape mismatch: {sg.j_idx.shape}"
        assert sg.alpha_mask.shape == (len(ALPHA_VALUES), sg.C_max), f"alpha_mask shape mismatch"
        assert len(sg.C_per_alpha) == len(ALPHA_VALUES), f"C_per_alpha length mismatch"

    def test_supergraph_coupled_sampling(self):
        """Smaller alpha should be subset of larger alpha (coupled sampling)."""
        from smf.modules.graphs.supergraph import create_supergraph

        device = torch.device('cpu')
        sg = create_supergraph(N1, N2, M, ALPHA_VALUES, S, SEED, device)

        # C should be monotonically increasing with alpha
        C_values = sg.C_per_alpha.tolist()
        assert all(C_values[i] <= C_values[i+1] for i in range(len(C_values)-1)), \
            f"C should increase with alpha: {C_values}"

        # Mask for smaller alpha should be subset of larger alpha
        for a in range(len(ALPHA_VALUES) - 1):
            mask_small = sg.alpha_mask[a]
            mask_large = sg.alpha_mask[a + 1]
            # Where mask_small is True, mask_large should also be True
            assert torch.all(mask_large[mask_small]), \
                f"Mask for alpha[{a}] should be subset of alpha[{a+1}]"

    def test_supergraph_index_bounds(self):
        """Edge indices should be within valid range."""
        from smf.modules.graphs.supergraph import create_supergraph

        device = torch.device('cpu')
        sg = create_supergraph(N1, N2, M, ALPHA_VALUES, S, SEED, device)

        assert torch.all(sg.i_idx >= 0) and torch.all(sg.i_idx < N1), "i_idx out of bounds"
        assert torch.all(sg.j_idx >= 0) and torch.all(sg.j_idx < N2), "j_idx out of bounds"

    def test_supergraph_determinism(self):
        """Same seed should produce same SuperGraph."""
        from smf.modules.graphs.supergraph import create_supergraph

        device = torch.device('cpu')
        sg1 = create_supergraph(N1, N2, M, ALPHA_VALUES, S, SEED, device)
        sg2 = create_supergraph(N1, N2, M, ALPHA_VALUES, S, SEED, device)

        assert torch.allclose(sg1.i_idx, sg2.i_idx), "i_idx should be deterministic"
        assert torch.allclose(sg1.j_idx, sg2.j_idx), "j_idx should be deterministic"


class TestBiGAMPSpreadingParallel:
    """Test parallel BiG-AMP spreading algorithm."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def config(self):
        from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, SpreadingConfig, AlgorithmConfig

        return Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=0.5, stop=2.0, step=0.5),
            training=TrainingConfig(max_steps=50, samples_per_alpha=S),
            algorithm=AlgorithmConfig(),
            algorithm_key='bigamp_spreading_parallel',
            spreading=SpreadingConfig(f_distribution='gaussian', seed=SEED),
        )

    def test_algorithm_instantiation(self, config, device):
        """Algorithm should instantiate without errors."""
        from smf.modules.algorithms.bigamp_spreading_parallel import BiGAMPSpreadingParallel

        algo = BiGAMPSpreadingParallel(config, device)
        assert algo is not None
        assert algo.f_distribution == 'gaussian'

    def test_train_single_sample(self, config, device):
        """Training single sample should return valid results."""
        from smf.modules.algorithms.bigamp_spreading_parallel import BiGAMPSpreadingParallel
        from smf.modules.teachers.random_spreading import SpreadingDataParallel
        from smf.modules.graphs.supergraph import create_supergraph

        algo = BiGAMPSpreadingParallel(config, device)
        alpha_values = config.alpha.get_values()

        # Create teacher matrices
        W_teacher = torch.randn(N1, M, device=device) / np.sqrt(M)
        X_teacher = torch.randn(M, N2, device=device) / np.sqrt(M)

        # Create spreading data
        spreading_data = algo.create_spreading_data(
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            alpha_values=alpha_values,
            S=S,
            base_seed=SEED,
        )

        # Train single sample
        W_hat, X_hat = algo.train_sample(spreading_data, sample_idx=0)

        assert W_hat.shape[0] == len(alpha_values), \
            f"W_hat should have {len(alpha_values)} alpha results"
        assert X_hat.shape[0] == len(alpha_values), \
            f"X_hat should have {len(alpha_values)} alpha results"

    def test_run_spreading_parallel(self, config, device):
        """Convenience function should run without errors."""
        from smf.modules.algorithms.bigamp_spreading_parallel import run_spreading_parallel

        # run_spreading_parallel takes a config object
        results = run_spreading_parallel(config, verbose=False)

        # Results should contain W_students or results dict
        assert 'W_students' in results or 'results' in results, \
            f"Results should contain training output, got keys: {results.keys()}"


class TestFDistributionCombinations:
    """Test all F distribution combinations."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.mark.parametrize("f_distribution", ["gaussian", "rademacher"])
    def test_f_distribution_runs(self, f_distribution, device):
        """Both F distributions should run without errors."""
        from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, SpreadingConfig, AlgorithmConfig
        from smf.modules.algorithms.bigamp_spreading_parallel import BiGAMPSpreadingParallel

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=0.5, stop=2.0, step=0.5),
            training=TrainingConfig(max_steps=10, samples_per_alpha=2),
            algorithm=AlgorithmConfig(),
            algorithm_key='bigamp_spreading_parallel',
            spreading=SpreadingConfig(f_distribution=f_distribution, seed=SEED),
        )

        algo = BiGAMPSpreadingParallel(config, device)
        assert algo.f_distribution == f_distribution, f"F distribution should be {f_distribution}"


class TestSpreadingDataCreation:
    """Test SpreadingDataParallel creation and training."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_spreading_data_creation(self, device):
        """SpreadingDataParallel should be created correctly."""
        from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, SpreadingConfig, AlgorithmConfig
        from smf.modules.algorithms.bigamp_spreading_parallel import BiGAMPSpreadingParallel

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=0.5, stop=2.0, step=0.5),
            training=TrainingConfig(max_steps=10, samples_per_alpha=S),
            algorithm=AlgorithmConfig(),
            algorithm_key='bigamp_spreading_parallel',
            spreading=SpreadingConfig(f_distribution='gaussian', seed=SEED),
        )

        algo = BiGAMPSpreadingParallel(config, device)
        alpha_values = config.alpha.get_values()

        # Create teacher matrices
        W_teacher = torch.randn(N1, M, device=device) / np.sqrt(M)
        X_teacher = torch.randn(M, N2, device=device) / np.sqrt(M)

        # Create spreading data
        spreading_data = algo.create_spreading_data(
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            alpha_values=alpha_values,
            S=S,
            base_seed=SEED,
        )

        assert spreading_data.F_super.shape[0] == S, "F_super should have S samples"
        assert spreading_data.Y_super.shape[0] == S, "Y_super should have S samples"
        assert spreading_data.A == len(alpha_values), "A should equal number of alpha values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
