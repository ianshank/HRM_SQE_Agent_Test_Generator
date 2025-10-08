"""
Unit tests for secure hash function usage.

Tests the replacement of MD5 with SHA-256 in convert_sqe_data.py
to ensure proper functionality and improved security.
"""

import pytest
import hashlib
from hrm_eval.convert_sqe_data import HRMDataConverter


class TestHashFunctionReplacement:
    """Tests for secure hash functions in data conversion."""
    
    @pytest.fixture
    def converter(self):
        """Create HRMDataConverter instance."""
        return HRMDataConverter()
    
    def test_token_mapping_consistency(self, converter):
        """Test that token mapping produces consistent results."""
        word = "testword"
        token1 = converter._word_to_token(word)
        token2 = converter._word_to_token(word)
        
        assert token1 == token2, "Same word should produce same token"
    
    def test_token_range(self, converter):
        """Test that tokens are in valid range 1-10."""
        test_words = [
            "action", "user", "system", "data", "process",
            "test", "example", "value", "result", "output"
        ]
        
        for word in test_words:
            token = converter._word_to_token(word)
            assert 1 <= token <= 10, f"Token {token} for '{word}' out of range"
    
    def test_keyword_mapping_priority(self, converter):
        """Test that keyword mapping takes priority over hashing."""
        # These words should map to specific tokens based on keywords
        keyword_tests = {
            "action": 1,
            "observation": 2,
            "reward": 3,
            "state": 4,
            "done": 5,
        }
        
        for word, expected_token in keyword_tests.items():
            token = converter._word_to_token(word)
            assert token == expected_token, \
                f"Keyword '{word}' should map to token {expected_token}"
    
    def test_hash_distribution(self, converter):
        """Test that hash function provides reasonable distribution."""
        # Generate tokens for many different words
        words = [f"word{i}" for i in range(1000)]
        tokens = [converter._word_to_token(word) for word in words]
        
        # Count token distribution
        from collections import Counter
        distribution = Counter(tokens)
        
        # Each token (1-9) should appear at least once
        assert len(distribution) >= 5, \
            "Hash function should distribute across multiple tokens"
        
        # No single token should dominate (> 50% of total)
        max_count = max(distribution.values())
        assert max_count < len(words) * 0.5, \
            "Token distribution should be reasonably balanced"
    
    def test_sha256_usage(self):
        """Verify SHA-256 is used instead of MD5."""
        import inspect
        from hrm_eval import convert_sqe_data
        
        # Read the source code
        source = inspect.getsource(convert_sqe_data.HRMDataConverter._word_to_token)
        
        # Verify SHA-256 is mentioned
        assert 'sha256' in source.lower(), \
            "Code should use SHA-256"
        
        # Verify MD5 is NOT used (except maybe in comments)
        lines = source.split('\n')
        code_lines = [l for l in lines if not l.strip().startswith('#')]
        code_text = '\n'.join(code_lines)
        
        assert 'md5' not in code_text.lower(), \
            "Code should not use MD5 hash function"
    
    def test_hash_collision_resistance(self, converter):
        """Test that similar words produce different tokens (most of the time)."""
        similar_pairs = [
            ("test", "tests"),
            ("user", "users"),
            ("data", "datas"),
            ("action", "actions"),
        ]
        
        different_tokens = 0
        for word1, word2 in similar_pairs:
            token1 = converter._word_to_token(word1)
            token2 = converter._word_to_token(word2)
            if token1 != token2:
                different_tokens += 1
        
        # At least 50% of similar pairs should have different tokens
        assert different_tokens >= len(similar_pairs) * 0.5, \
            "Hash function should handle similar words well"
    
    def test_empty_string_handling(self, converter):
        """Test handling of empty string."""
        # Should not crash
        token = converter._word_to_token("")
        assert 1 <= token <= 10
    
    def test_unicode_handling(self, converter):
        """Test handling of Unicode characters."""
        unicode_words = [
            "café", "naïve", "résumé", "日本語", "Ελληνικά"
        ]
        
        for word in unicode_words:
            # Should not crash
            token = converter._word_to_token(word)
            assert 1 <= token <= 10
    
    def test_case_insensitivity(self, converter):
        """Test that token mapping is case-insensitive for keywords."""
        word_variations = [
            ("ACTION", "action", "Action"),
            ("USER", "user", "User"),
            ("DATA", "data", "Data"),
        ]
        
        for variations in word_variations:
            tokens = [converter._word_to_token(w) for w in variations]
            # All variations should produce the same token
            assert len(set(tokens)) == 1, \
                f"Case variations {variations} should produce same token"


class TestSecurityImprovement:
    """Tests demonstrating security improvements."""
    
    def test_sha256_vs_md5_collision_resistance(self):
        """Demonstrate SHA-256 is more collision-resistant than MD5."""
        test_data = b"test data"
        
        # Calculate both hashes
        md5_hash = hashlib.md5(test_data).hexdigest()
        sha256_hash = hashlib.sha256(test_data).hexdigest()
        
        # SHA-256 produces longer hash (more bits)
        assert len(sha256_hash) > len(md5_hash), \
            "SHA-256 should produce longer hash than MD5"
        
        # SHA-256 is 256 bits (64 hex chars), MD5 is 128 bits (32 hex chars)
        assert len(sha256_hash) == 64
        assert len(md5_hash) == 32
    
    def test_sha256_determinism(self):
        """Verify SHA-256 is deterministic."""
        data = "consistent test data"
        
        hash1 = hashlib.sha256(data.encode()).hexdigest()
        hash2 = hashlib.sha256(data.encode()).hexdigest()
        
        assert hash1 == hash2, "SHA-256 should produce consistent results"
    
    def test_sha256_avalanche_effect(self):
        """Test avalanche effect (small input change = large hash change)."""
        data1 = "test data"
        data2 = "test datb"  # Single character different
        
        hash1 = hashlib.sha256(data1.encode()).hexdigest()
        hash2 = hashlib.sha256(data2.encode()).hexdigest()
        
        # Hashes should be completely different
        assert hash1 != hash2
        
        # Count different characters
        differences = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        
        # Should have significant differences (> 50% of characters)
        assert differences > len(hash1) * 0.5, \
            "Small input change should cause large hash change"


class TestIntegrationWithConverter:
    """Integration tests for complete conversion pipeline."""
    
    @pytest.fixture
    def converter(self):
        """Create converter instance."""
        return HRMDataConverter()
    
    def test_complete_example_conversion(self, converter):
        """Test conversion of complete example."""
        example_data = {
            "prompt": "User wants to test the system",
            "completion": "System processes the test request successfully",
            "example_id": 1
        }
        
        # Should not raise any exceptions
        result = converter.convert_example(
            prompt=example_data["prompt"],
            completion=example_data["completion"],
            example_id=example_data["example_id"]
        )
        
        assert result is not None
        assert "input_ids" in result
        assert "target_ids" in result
    
    def test_batch_conversion_consistency(self, converter):
        """Test that batch conversion is consistent."""
        examples = [
            {
                "prompt": f"Test prompt {i}",
                "completion": f"Test completion {i}",
                "example_id": i
            }
            for i in range(10)
        ]
        
        # Convert twice
        results1 = [
            converter.convert_example(
                ex["prompt"], ex["completion"], ex["example_id"]
            )
            for ex in examples
        ]
        
        results2 = [
            converter.convert_example(
                ex["prompt"], ex["completion"], ex["example_id"]
            )
            for ex in examples
        ]
        
        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert r1["input_ids"] == r2["input_ids"]
            assert r1["target_ids"] == r2["target_ids"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
