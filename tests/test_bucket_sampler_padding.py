"""Tests for bucket sampler padding of undersized buckets."""

import pytest
from collections import Counter


class TestBucketPadding:
    def _make_sampler(self, bucket_assignments, batch_size=2, drop_last=True):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        return BucketBatchSampler(
            bucket_assignments=bucket_assignments,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=False,
            seed=0,
        )

    def test_single_image_bucket_is_not_dropped(self):
        """A bucket with only 1 image and batch_size=2 must still produce 1 batch."""
        # 4 images: 3 in bucket (512,512), 1 in bucket (768,768)
        assignments = [(512, 512), (512, 512), (512, 512), (768, 768)]
        sampler = self._make_sampler(assignments, batch_size=2)

        batches = list(sampler)
        # The lonely (768,768) image MUST appear somewhere
        all_indices_used = [i for batch in batches for i in batch]
        assert 3 in all_indices_used, "Index 3 (lonely image) was dropped"

    def test_undersized_bucket_padded_by_repetition(self):
        """Bucket with 1 image and batch_size=4: yields batch [0,0,0,0]."""
        assignments = [(768, 768)]
        sampler = self._make_sampler(assignments, batch_size=4)
        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == [0, 0, 0, 0]

    def test_padded_batch_size_matches_batch_size(self):
        """All emitted batches have len == batch_size when drop_last=True."""
        assignments = [
            (512, 512), (512, 512), (512, 512), (512, 512),  # 4 normal
            (768, 768),  # 1 alone
            (1024, 1024), (1024, 1024),  # 2 — still < 3
        ]
        sampler = self._make_sampler(assignments, batch_size=3, drop_last=True)
        batches = list(sampler)
        for batch in batches:
            assert len(batch) == 3

    def test_total_batches_count_correct(self):
        """__len__ reports the actual number of yielded batches."""
        # 5 in bucket A (batch_size=2 → 2 batches with drop_last, last 1 dropped),
        # 1 in bucket B (padded to 1 batch),
        # 4 in bucket C (2 batches exact).
        # Expected: 2 + 1 + 2 = 5 batches.
        assignments = (
            [(512, 512)] * 5
            + [(768, 768)] * 1
            + [(1024, 1024)] * 4
        )
        sampler = self._make_sampler(assignments, batch_size=2, drop_last=True)
        batches = list(sampler)
        assert len(batches) == 5
        assert len(sampler) == 5

    def test_no_padding_when_bucket_meets_batch_size(self):
        """Bucket with exactly batch_size images yields one batch with no repetition."""
        assignments = [(512, 512), (512, 512)]
        sampler = self._make_sampler(assignments, batch_size=2)
        batches = list(sampler)
        assert len(batches) == 1
        # No index should appear twice in a single batch
        assert len(set(batches[0])) == 2

    def test_lonely_image_indices_always_appear(self):
        """Undersized buckets must contribute their images even with drop_last=True.

        Normal buckets still honor drop_last (so the unpaired tail of a
        multi-image bucket is allowed to drop). This test asserts only the
        new behavior: the unique image of a 1-image bucket is never dropped.
        """
        assignments = [
            (512, 512), (512, 512),         # 2 → 1 batch
            (768, 768),                      # 1 lonely → padded
            (1024, 1024),                    # 1 lonely → padded
            (640, 640), (640, 640),          # 2 → 1 batch
        ]
        sampler = self._make_sampler(assignments, batch_size=2, drop_last=True)
        batches = list(sampler)
        seen = set(i for batch in batches for i in batch)
        # Indices 2 and 3 are the lonely ones — they must survive.
        assert 2 in seen, "Lonely image at index 2 was dropped"
        assert 3 in seen, "Lonely image at index 3 was dropped"
