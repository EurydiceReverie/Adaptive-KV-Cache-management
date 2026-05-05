"""
Unit Tests — Phase 5 Learned Memory + Clustering Algorithm
===========================================================
Tests run WITHOUT a real model (uses synthetic tensors).
Run with:
    python -m pytest tests/test_phase5_and_clustering.py -v
    python tests/test_phase5_and_clustering.py  (standalone)
"""
from __future__ import annotations
import sys, os, math, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


# ─── helpers to build fake KV caches ────────────────────────────────────────

def make_fake_cache(num_layers=4, num_heads=4, seq_len=128, head_dim=32,
                    dtype=torch.float32) -> tuple:
    """Create a synthetic KV cache tuple for testing."""
    return tuple(
        (
            torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype),
            torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype),
        )
        for _ in range(num_layers)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. K-Means++ Clustering Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKMeansPlusPlus(unittest.TestCase):

    def test_init_returns_correct_shape(self):
        from kv_cache_compression.cache.cluster import _kmeans_plus_plus_init
        features = torch.randn(100, 16)
        centers = _kmeans_plus_plus_init(features, num_clusters=8)
        self.assertEqual(centers.shape, (8, 16))

    def test_init_centers_are_from_data(self):
        """All initialised centroids must come from the input data."""
        from kv_cache_compression.cache.cluster import _kmeans_plus_plus_init
        features = torch.randn(50, 8)
        centers = _kmeans_plus_plus_init(features, num_clusters=5)
        # Each center should match exactly one row in features
        for ci in range(centers.shape[0]):
            dists = (features - centers[ci]).norm(dim=-1)
            self.assertTrue(dists.min().item() < 1e-5,
                            f"Center {ci} not found in input data")

    def test_simple_kmeans_returns_assignments(self):
        from kv_cache_compression.cache.cluster import _simple_kmeans
        features = torch.randn(64, 16)
        assignments = _simple_kmeans(features, num_clusters=8)
        self.assertEqual(assignments.shape[0], 64)
        self.assertTrue(assignments.min() >= 0)
        self.assertTrue(assignments.max() < 8)

    def test_kmeans_fewer_points_than_clusters(self):
        """When seq_len <= clusters, should return identity assignment."""
        from kv_cache_compression.cache.cluster import _simple_kmeans
        features = torch.randn(5, 16)
        result = _simple_kmeans(features, num_clusters=10)
        # Should return arange(5)
        self.assertEqual(len(result), 5)

    def test_kmeans_cluster_coverage(self):
        """All data points should be assigned to a cluster."""
        from kv_cache_compression.cache.cluster import _simple_kmeans
        features = torch.randn(100, 8)
        assignments = _simple_kmeans(features, num_clusters=6)
        # Every index 0..99 must appear
        self.assertEqual(assignments.shape[0], 100)
        unique_clusters = assignments.unique()
        self.assertGreaterEqual(len(unique_clusters), 1)

    def test_kmeans_separable_clusters(self):
        """Well-separated clusters should be recovered correctly."""
        from kv_cache_compression.cache.cluster import _simple_kmeans
        torch.manual_seed(42)
        # 3 clearly separated groups
        g1 = torch.randn(30, 4) + torch.tensor([10., 0., 0., 0.])
        g2 = torch.randn(30, 4) + torch.tensor([0., 10., 0., 0.])
        g3 = torch.randn(30, 4) + torch.tensor([0., 0., 10., 0.])
        features = torch.cat([g1, g2, g3], dim=0)  # [90, 4]
        assignments = _simple_kmeans(features, num_clusters=3)
        # Each group should map to one unique cluster
        c1 = set(assignments[:30].tolist())
        c2 = set(assignments[30:60].tolist())
        c3 = set(assignments[60:].tolist())
        self.assertEqual(len(c1), 1, f"Group 1 spans clusters: {c1}")
        self.assertEqual(len(c2), 1, f"Group 2 spans clusters: {c2}")
        self.assertEqual(len(c3), 1, f"Group 3 spans clusters: {c3}")
        self.assertEqual(len(c1 | c2 | c3), 3, "Should have 3 distinct cluster IDs")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Learned Memory Helper Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLearnedMemoryHelpers(unittest.TestCase):

    def test_kmeans_centroids_shape(self):
        from kv_cache_compression.cache.learned_memory import _kmeans_centroids
        features = torch.randn(80, 20)
        centroids = _kmeans_centroids(features, n_clusters=8)
        self.assertEqual(centroids.shape, (8, 20))

    def test_kmeans_centroids_padding_when_few_points(self):
        from kv_cache_compression.cache.learned_memory import _kmeans_centroids
        features = torch.randn(3, 16)
        centroids = _kmeans_centroids(features, n_clusters=8)
        self.assertEqual(centroids.shape, (8, 16))

    def test_kv_to_features_shape(self):
        from kv_cache_compression.cache.learned_memory import _kv_to_features
        cache = make_fake_cache(num_layers=4, num_heads=4, seq_len=64, head_dim=16)
        features = _kv_to_features(cache)
        # Should be [64, 2*16] = [64, 32]
        self.assertEqual(features.shape, (64, 32))

    def test_attention_output_approx_shape(self):
        from kv_cache_compression.cache.learned_memory import _attention_output_approx
        q = torch.randn(1, 4, 1, 16)
        k = torch.randn(1, 4, 32, 16)
        v = torch.randn(1, 4, 32, 16)
        out = _attention_output_approx(q, k, v)
        self.assertEqual(out.shape, (1, 4, 1, 16))

    def test_attention_output_approx_values_finite(self):
        from kv_cache_compression.cache.learned_memory import _attention_output_approx
        q = torch.randn(1, 2, 1, 8)
        k = torch.randn(1, 2, 10, 8)
        v = torch.randn(1, 2, 10, 8)
        out = _attention_output_approx(q, k, v)
        self.assertTrue(torch.isfinite(out).all())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LearnedMemoryPolicy Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLearnedMemoryPolicy(unittest.TestCase):

    def _make_policy(self, n_memory=8, optim_steps=3, always_keep_last=16):
        from kv_cache_compression.cache.learned_memory import LearnedMemoryPolicy
        return LearnedMemoryPolicy(
            n_memory=n_memory,
            optim_steps=optim_steps,
            lr=1e-2,
            always_keep_last=always_keep_last,
        )

    def test_output_shape_correct(self):
        """Compressed cache must have n_memory + always_keep_last tokens."""
        policy = self._make_policy(n_memory=8, always_keep_last=16)
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=64, head_dim=16)
        outcome = policy.compress(cache)
        self.assertEqual(outcome.compressed_tokens, 8 + 16)

    def test_num_layers_preserved(self):
        """Compressed cache must have same number of layers."""
        policy = self._make_policy()
        cache = make_fake_cache(num_layers=4, num_heads=2, seq_len=64, head_dim=16)
        outcome = policy.compress(cache)
        self.assertEqual(len(outcome.past_key_values), 4)

    def test_skips_when_already_small(self):
        """If cache already fits budget, compression should be skipped."""
        policy = self._make_policy(n_memory=32, always_keep_last=32)
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=20, head_dim=8)
        outcome = policy.compress(cache)
        self.assertTrue(outcome.metadata.get("skipped", False))
        self.assertEqual(outcome.original_tokens, outcome.compressed_tokens)

    def test_compression_ratio_below_one(self):
        """Compression must reduce token count."""
        policy = self._make_policy(n_memory=8, always_keep_last=8, optim_steps=2)
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=128, head_dim=16)
        outcome = policy.compress(cache)
        self.assertLess(outcome.compression_ratio, 1.0)

    def test_output_dtype_preserved(self):
        """Output KV tensors must keep the same dtype as input."""
        policy = self._make_policy(n_memory=8, always_keep_last=8, optim_steps=2)
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=64, head_dim=16,
                                dtype=torch.float32)
        outcome = policy.compress(cache)
        for key, val in outcome.past_key_values:
            self.assertEqual(key.dtype, torch.float32)
            self.assertEqual(val.dtype, torch.float32)

    def test_metadata_contains_expected_keys(self):
        """Outcome metadata must contain n_memory, optim_steps, etc."""
        policy = self._make_policy(n_memory=8, always_keep_last=8, optim_steps=2)
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=64, head_dim=16)
        outcome = policy.compress(cache)
        for key in ["n_memory", "optim_steps", "lr", "always_keep_last"]:
            self.assertIn(key, outcome.metadata, f"Missing key: {key}")

    def test_no_gradient_leak(self):
        """Compressed cache tensors must be detached (no grad)."""
        policy = self._make_policy(n_memory=8, always_keep_last=8, optim_steps=3)
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=64, head_dim=16)
        outcome = policy.compress(cache)
        for key, val in outcome.past_key_values:
            self.assertFalse(key.requires_grad, "Key tensor has grad — not detached!")
            self.assertFalse(val.requires_grad, "Val tensor has grad — not detached!")

    def test_kmeans_init_vs_random_init(self):
        """K-Means init should produce valid output same as random init."""
        from kv_cache_compression.cache.learned_memory import LearnedMemoryPolicy
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=64, head_dim=16)
        for use_init in [True, False]:
            pol = LearnedMemoryPolicy(n_memory=8, optim_steps=2,
                                      always_keep_last=8, use_kmeans_init=use_init)
            outcome = pol.compress(cache)
            self.assertEqual(outcome.compressed_tokens, 16)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MemoryTokenAdapter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryTokenAdapter(unittest.TestCase):

    def _make_adapter(self, head_dim=16, n_memory=8):
        from kv_cache_compression.cache.learned_memory import MemoryTokenAdapter
        return MemoryTokenAdapter(head_dim=head_dim, n_memory=n_memory,
                                  hidden_dim=64, num_layers=2)

    def test_forward_output_shapes(self):
        adapter = self._make_adapter(head_dim=16, n_memory=8)
        key = torch.randn(1, 4, 32, 16)
        val = torch.randn(1, 4, 32, 16)
        mem_key, mem_val = adapter(key, val)
        self.assertEqual(mem_key.shape, (1, 4, 8, 16))
        self.assertEqual(mem_val.shape, (1, 4, 8, 16))

    def test_loss_is_positive_scalar(self):
        adapter = self._make_adapter(head_dim=16, n_memory=8)
        key = torch.randn(1, 4, 32, 16)
        val = torch.randn(1, 4, 32, 16)
        loss = adapter.compute_loss(key, val)
        self.assertEqual(loss.shape, ())       # scalar
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_loss_decreases_with_training(self):
        """A few gradient steps should reduce the loss."""
        torch.manual_seed(0)
        adapter = self._make_adapter(head_dim=16, n_memory=4)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        key = torch.randn(1, 2, 32, 16)
        val = torch.randn(1, 2, 32, 16)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = adapter.compute_loss(key, val)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0],
                        f"Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}")

    def test_save_and_load(self):
        """Save and load cycle must produce identical outputs."""
        adapter = self._make_adapter(head_dim=16, n_memory=8)
        key = torch.randn(1, 4, 32, 16)
        val = torch.randn(1, 4, 32, 16)
        with torch.no_grad():
            out_before = adapter(key, val)[0].clone()

        # Save to local tmp/ folder (not system temp)
        os.makedirs("tmp", exist_ok=True)
        path = os.path.join("tmp", "test_adapter_save_load.pt")
        try:
            adapter.save(path)
            from kv_cache_compression.cache.learned_memory import MemoryTokenAdapter
            loaded = MemoryTokenAdapter.load(path, head_dim=16, n_memory=8,
                                             hidden_dim=64, num_layers=2)
            with torch.no_grad():
                out_after = loaded(key, val)[0]
            self.assertTrue(torch.allclose(out_before, out_after),
                            "Loaded adapter gives different output")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KVCacheInspector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKVCacheInspector(unittest.TestCase):

    def test_sequence_length(self):
        from kv_cache_compression.cache.kv_cache import KVCacheInspector
        cache = make_fake_cache(seq_len=77)
        self.assertEqual(KVCacheInspector.sequence_length(cache), 77)

    def test_num_layers(self):
        from kv_cache_compression.cache.kv_cache import KVCacheInspector
        cache = make_fake_cache(num_layers=6)
        self.assertEqual(KVCacheInspector.num_layers(cache), 6)

    def test_total_bytes(self):
        from kv_cache_compression.cache.kv_cache import KVCacheInspector
        # 2 layers, 2 heads, 10 tokens, 8 head_dim, float32 (4 bytes)
        cache = make_fake_cache(num_layers=2, num_heads=2, seq_len=10, head_dim=8,
                                dtype=torch.float32)
        expected = 2 * 2 * (1 * 2 * 10 * 8 * 4)   # layers * (key+val) * numel * bytes
        self.assertEqual(KVCacheInspector.total_bytes(cache), expected)

    def test_empty_cache(self):
        from kv_cache_compression.cache.kv_cache import KVCacheInspector
        self.assertEqual(KVCacheInspector.sequence_length(()), 0)
        self.assertEqual(KVCacheInspector.num_layers(()), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CompressionOutcome Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompressionOutcome(unittest.TestCase):

    def test_compression_ratio(self):
        from kv_cache_compression.cache.kv_cache import CompressionOutcome
        cache = make_fake_cache(num_layers=1, seq_len=10)
        outcome = CompressionOutcome(
            policy_name="test",
            past_key_values=cache,
            original_tokens=100,
            compressed_tokens=25,
            kept_token_indices=list(range(25)),
        )
        self.assertAlmostEqual(outcome.compression_ratio, 0.25)

    def test_token_reduction(self):
        from kv_cache_compression.cache.kv_cache import CompressionOutcome
        cache = make_fake_cache(num_layers=1, seq_len=10)
        outcome = CompressionOutcome(
            policy_name="test",
            past_key_values=cache,
            original_tokens=100,
            compressed_tokens=40,
            kept_token_indices=list(range(40)),
        )
        self.assertEqual(outcome.token_reduction, 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Recency and Attention Prune Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPruneFunctions(unittest.TestCase):

    def test_build_recency_indices_basic(self):
        from kv_cache_compression.cache.prune import build_recency_indices
        result = build_recency_indices(seq_len=100, keep_last_tokens=20)
        self.assertEqual(result, list(range(80, 100)))

    def test_build_recency_indices_exceeds_seq(self):
        from kv_cache_compression.cache.prune import build_recency_indices
        result = build_recency_indices(seq_len=10, keep_last_tokens=50)
        self.assertEqual(result, list(range(10)))

    def test_build_attention_indices_count(self):
        from kv_cache_compression.cache.prune import build_attention_indices
        scores = torch.rand(1, 50)
        indices = build_attention_indices(scores, keep_last_tokens=10, keep_top_tokens=5)
        # Should keep at most keep_last + keep_top = 15 unique indices
        self.assertLessEqual(len(indices), 15)
        self.assertGreaterEqual(len(indices), 1)

    def test_attention_indices_sorted(self):
        from kv_cache_compression.cache.prune import build_attention_indices
        scores = torch.rand(1, 50)
        indices = build_attention_indices(scores, keep_last_tokens=10, keep_top_tokens=5)
        self.assertEqual(indices, sorted(indices))


if __name__ == "__main__":
    print("Running Phase 5 + Clustering unit tests...")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestKMeansPlusPlus,
        TestLearnedMemoryHelpers,
        TestLearnedMemoryPolicy,
        TestMemoryTokenAdapter,
        TestKVCacheInspector,
        TestCompressionOutcome,
        TestPruneFunctions,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
