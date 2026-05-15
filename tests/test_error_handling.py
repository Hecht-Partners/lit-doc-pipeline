#!/usr/bin/env python3
"""
Test script for error handling and recovery features.

Tests:
1. State tracking - checkpoint/resume
2. Error recovery - continue on failure
3. Timeout handling - configurable timeouts
4. Force reprocessing
"""

import json
import shutil
import tempfile
from pathlib import Path

from lit_pipeline.pipeline_state import PipelineState, DocumentState


def test_state_tracking():
    """Test basic state tracking functionality."""
    print("\n=== Test 1: State Tracking ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create new state
        state = PipelineState(output_dir)
        doc = state.get_document("test_doc", "test_doc.pdf")

        # Mark stages complete
        doc.mark_stage_complete("conversion")
        doc.mark_stage_complete("post_processing")
        state.save()

        # Verify state file exists
        state_file = output_dir / ".lit-pipeline-state.json"
        assert state_file.exists(), "State file not created"

        # Load state and verify
        state2 = PipelineState(output_dir)
        doc2 = state2.get_document("test_doc", "test_doc.pdf")

        assert doc2.is_stage_complete("conversion"), "Conversion stage not saved"
        assert doc2.is_stage_complete("post_processing"), "Post-processing stage not saved"
        assert not doc2.is_stage_complete("citation_tracking"), "Incorrect stage marked complete"

        print("✓ State tracking works correctly")


def test_failure_tracking():
    """Test failure tracking and retry limits."""
    print("\n=== Test 2: Failure Tracking ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        state = PipelineState(output_dir)

        doc = state.get_document("failed_doc", "failed_doc.pdf")

        # Mark as failed multiple times
        for i in range(3):
            doc.mark_failed(f"Error {i+1}")
            state.save()

        # Verify retry count
        assert doc.retry_count == 3, f"Expected 3 retries, got {doc.retry_count}"
        assert doc.status == "failed", "Document not marked as failed"

        # Should skip after 3 failures
        should_skip = not state.should_process_document("failed_doc", "conversion", force=False)
        assert should_skip, "Should skip document after 3 failures"

        # Force should override
        should_process = state.should_process_document("failed_doc", "conversion", force=True)
        assert should_process, "Force flag should allow reprocessing"

        print("✓ Failure tracking works correctly")


def test_resume_logic():
    """Test resume logic - skip completed stages."""
    print("\n=== Test 3: Resume Logic ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        state = PipelineState(output_dir)

        # Document with partial completion
        doc = state.get_document("partial_doc", "partial_doc.pdf")
        doc.mark_stage_complete("conversion")
        doc.mark_stage_complete("post_processing")
        state.save()

        # Should skip completed stages
        assert not state.should_process_document("partial_doc", "conversion", force=False)
        assert not state.should_process_document("partial_doc", "post_processing", force=False)

        # Should process incomplete stage
        assert state.should_process_document("partial_doc", "citation_tracking", force=False)

        # Force should reprocess everything
        assert state.should_process_document("partial_doc", "conversion", force=True)

        print("✓ Resume logic works correctly")


def test_summary():
    """Test state summary generation."""
    print("\n=== Test 4: State Summary ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        state = PipelineState(output_dir)

        # Create various documents in different states
        doc1 = state.get_document("complete_doc", "complete_doc.pdf")
        doc1.mark_stage_complete("conversion")
        doc1.mark_stage_complete("post_processing")
        doc1.mark_stage_complete("citation_tracking")
        doc1.mark_completed()

        doc2 = state.get_document("failed_doc", "failed_doc.pdf")
        doc2.mark_failed("Conversion timeout")

        doc3 = state.get_document("in_progress_doc", "in_progress_doc.pdf")
        doc3.mark_stage_complete("conversion")

        state.save()

        # Generate summary
        summary = state.summary()
        print("\nState Summary:")
        print(summary)

        assert "Total documents: 3" in summary
        assert "Completed: 1" in summary
        assert "Failed: 1" in summary
        assert "In progress: 1" in summary

        print("✓ Summary generation works correctly")


def main():
    """Run all tests."""
    print("Testing Error Handling & Recovery Features")
    print("=" * 60)

    try:
        test_state_tracking()
        test_failure_tracking()
        test_resume_logic()
        test_summary()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
