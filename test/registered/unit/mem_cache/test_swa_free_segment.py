"""SWA segment frees must not fall back to token-wise torch.unique."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 4
SIZE = 512


def _allocator():
    pool = MagicMock(spec=BaseSWAKVPool)
    pool.full_kv_pool = None
    pool.swa_kv_pool = None
    return SWATokenToKVPoolAllocator(
        size=SIZE,
        size_swa=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )


def _alloc_row(allocator, num_pages):
    num_tokens = num_pages * PAGE_SIZE
    full = allocator.full_attn_allocator.alloc(num_tokens)
    swa = allocator.swa_attn_allocator.alloc(num_tokens)
    assert full is not None and swa is not None
    allocator.set_full_to_swa_mapping(full, swa)
    return full


class TestSWAFreeSegment(unittest.TestCase):
    def test_96_way_group_never_materializes_token_unique(self):
        allocator = _allocator()
        rows = [_alloc_row(allocator, 1) for _ in range(96)]
        allocator.free_group_begin()
        for row in rows:
            allocator.free_segment(row, start_pos=0)
        with patch.object(torch, "unique", side_effect=AssertionError("unique called")):
            allocator.free_group_end()
        self.assertEqual(allocator.full_available_size(), SIZE)
        self.assertEqual(allocator.swa_available_size(), SIZE)

    def test_grouped_free_uses_page_representatives(self):
        allocator = _allocator()
        row = _alloc_row(allocator, 3)
        original_row = row.clone()
        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()

        allocator.free_group_begin()
        allocator.free_segment(row, start_pos=0)
        row.zero_()
        with patch.object(torch, "unique", side_effect=AssertionError("unique called")):
            allocator.free_group_end()

        self.assertEqual(allocator.full_available_size(), full_before + 3 * PAGE_SIZE)
        self.assertEqual(allocator.swa_available_size(), swa_before + 3 * PAGE_SIZE)
        self.assertTrue(
            torch.equal(
                allocator.full_to_swa_index_mapping[original_row],
                torch.zeros_like(original_row),
            )
        )

    def test_unaligned_segment_releases_each_touched_page_once(self):
        allocator = _allocator()
        row = _alloc_row(allocator, 3)
        touched = row[: 2 * PAGE_SIZE].clone()
        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()

        with patch.object(torch, "unique", side_effect=AssertionError("unique called")):
            allocator.free_segment(row[1:7], start_pos=1)

        self.assertEqual(allocator.full_available_size(), full_before + 2 * PAGE_SIZE)
        self.assertEqual(allocator.swa_available_size(), swa_before + 2 * PAGE_SIZE)
        self.assertTrue(
            torch.equal(
                allocator.full_to_swa_index_mapping[touched],
                torch.zeros_like(touched),
            )
        )

    def test_segment_skips_already_tombstoned_swa_page(self):
        allocator = _allocator()
        row = _alloc_row(allocator, 3)
        allocator.free_swa(row[:PAGE_SIZE])
        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()

        allocator.free_group_begin()
        allocator.free_segment(row, start_pos=0)
        with patch.object(torch, "unique", side_effect=AssertionError("unique called")):
            allocator.free_group_end()

        self.assertEqual(allocator.full_available_size(), full_before + 3 * PAGE_SIZE)
        self.assertEqual(allocator.swa_available_size(), swa_before + 2 * PAGE_SIZE)

    def test_swa_only_segment_is_sync_free(self):
        allocator = _allocator()
        row = _alloc_row(allocator, 2)
        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()

        allocator.free_group_begin()
        allocator.free_swa_segment(row, start_pos=0)
        row.zero_()
        with patch.object(torch, "unique", side_effect=AssertionError("unique called")):
            allocator.free_group_end()

        self.assertEqual(allocator.full_available_size(), full_before)
        self.assertEqual(allocator.swa_available_size(), swa_before + 2 * PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
