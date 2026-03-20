"""Unit tests for data/explore_dataset.py helper functions."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from data.explore_dataset import check_arguments_encoding, collect_stats


class TestCheckArgumentsEncoding:
    """Tests for check_arguments_encoding()."""

    def test_returns_dict_when_arguments_is_dict(self):
        """Arguments already a dict should be returned as-is and flag double_encoded=False."""
        answers = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        result = check_arguments_encoding(answers)
        assert result["double_encoded"] is False
        assert result["arguments"] == {"location": "NYC"}

    def test_returns_parsed_dict_when_arguments_is_string(self):
        """Arguments as a JSON string should be parsed and flag double_encoded=True."""
        answers = [{"name": "get_weather", "arguments": '{"location": "NYC"}'}]
        result = check_arguments_encoding(answers)
        assert result["double_encoded"] is True
        assert result["arguments"] == {"location": "NYC"}

    def test_returns_none_when_answers_empty(self):
        """Empty answers list should return None."""
        result = check_arguments_encoding([])
        assert result is None

    def test_returns_none_when_no_arguments_key(self):
        """Answer without 'arguments' key should return None."""
        answers = [{"name": "get_weather"}]
        result = check_arguments_encoding(answers)
        assert result is None


class TestCollectStats:
    """Tests for collect_stats()."""

    def _make_row(self, query, tools, answers):
        """Build a fake dataset row with JSON-encoded tools and answers."""
        return {
            "query": query,
            "tools": json.dumps(tools),
            "answers": json.dumps(answers),
        }

    def test_counts_query_lengths(self):
        """Query lengths should be recorded for every row."""
        rows = [
            self._make_row("hi", [], [{"name": "f", "arguments": {}}]),
            self._make_row("hello there", [], [{"name": "f", "arguments": {}}]),
        ]
        stats = collect_stats(rows)
        assert stats["query_lens"] == [2, 11]

    def test_counts_tools_and_calls(self):
        """Tools and call counts should reflect parsed list lengths."""
        rows = [
            self._make_row(
                "query",
                [{"name": "a"}, {"name": "b"}],
                [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}],
            )
        ]
        stats = collect_stats(rows)
        assert stats["tools_counts"] == [2]
        assert stats["calls_counts"] == [2]

    def test_records_parse_failure_on_invalid_json(self):
        """Rows with malformed JSON should increment parse_fails without crashing."""
        rows = [{"query": "q", "tools": "not-json", "answers": "[]"}]
        stats = collect_stats(rows)
        assert stats["parse_fails"] == 1
        assert stats["tools_counts"] == []
        assert stats["calls_counts"] == []
        assert stats["query_lens"] == [1]

    def test_all_rows_fail_parse(self):
        """All rows failing should produce empty tools/calls lists and non-zero parse_fails."""
        rows = [
            {"query": "q1", "tools": "bad", "answers": "bad"},
            {"query": "q2", "tools": "bad", "answers": "bad"},
        ]
        stats = collect_stats(rows)
        assert stats["parse_fails"] == 2
        assert stats["tools_counts"] == []
        assert stats["calls_counts"] == []

    def test_parallel_call_detection(self):
        """Rows with more than one answer should be counted as parallel calls."""
        rows = [
            self._make_row("q", [], [{"name": "a", "arguments": {}}]),
            self._make_row("q", [], [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]),
        ]
        stats = collect_stats(rows)
        parallel = sum(1 for n in stats["calls_counts"] if n > 1)
        assert parallel == 1

    def test_empty_dataset(self):
        """Empty input should return zeroed stats without raising."""
        stats = collect_stats([])
        assert stats["parse_fails"] == 0
        assert stats["query_lens"] == []
        assert stats["tools_counts"] == []
        assert stats["calls_counts"] == []
