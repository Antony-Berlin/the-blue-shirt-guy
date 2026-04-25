"""Unit tests for the multi-grader system."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.gen_env.server.tool_graders import (
    ContentTypeDetector,
    ContentType,
    CodeSyntaxGrader,
    CodeStyleGrader,
    TestResultGrader,
    ErrorExplanationGrader,
    SearchRelevanceGrader,
    DocQualityGrader,
    ReasoningGrader,
    RedundancyGrader,
    ErrorPropagationGrader,
    GraderRouter,
)


# ---------------------------------------------------------------------------
# ContentTypeDetector
# ---------------------------------------------------------------------------

class TestContentTypeDetector:
    def setup_method(self):
        self.d = ContentTypeDetector()

    def test_detects_test_passed(self):
        assert self.d.detect("PASSED: All test cases passed.") == ContentType.TEST_RESULT

    def test_detects_test_failed(self):
        assert self.d.detect("FAILED:\nAssertionError at line 3") == ContentType.TEST_RESULT

    def test_detects_error_traceback(self):
        tb = "Traceback (most recent call last):\n  File x.py, line 5\nTypeError: bad type"
        assert self.d.detect(tb) == ContentType.ERROR_TRACEBACK

    def test_detects_code(self):
        code = "def add(a, b):\n    return a + b\n"
        assert self.d.detect(code) == ContentType.CODE

    def test_detects_search_result(self):
        result = "1. example one\n2. example two\n3. example three\n"
        assert self.d.detect(result) == ContentType.SEARCH_RESULT

    def test_detects_documentation(self):
        doc = (
            "Help on built-in function sorted:\n\n"
            "DESCRIPTION\n    sorted(iterable, /, *, key=None, reverse=False)\n"
            "    Return a new list containing all items from the iterable in ascending order.\n"
        )
        assert self.d.detect(doc) == ContentType.DOCUMENTATION

    def test_detects_text_fallback(self):
        assert self.d.detect("The function looks correct overall.") == ContentType.TEXT

    def test_empty_string_is_text(self):
        assert self.d.detect("") == ContentType.TEXT


# ---------------------------------------------------------------------------
# CodeSyntaxGrader
# ---------------------------------------------------------------------------

class TestCodeSyntaxGrader:
    def setup_method(self):
        self.g = CodeSyntaxGrader()

    def test_valid_code_scores_high(self):
        code = "def add(a, b):\n    return a + b\n"
        assert self.g.grade(code) >= 0.8

    def test_syntax_error_scores_zero(self):
        assert self.g.grade("def foo(:\n    pass") == 0.0

    def test_placeholder_pass_penalised(self):
        code = "def add(a, b):\n    pass\n"
        score = self.g.grade(code)
        # pass-only body is penalised vs real implementation (which scores 1.0)
        assert score < 0.9

    def test_not_implemented_penalised(self):
        code = "def add(a, b):\n    raise NotImplementedError()\n"
        score = self.g.grade(code)
        assert score < 0.9

    def test_no_function_penalised(self):
        code = "x = 1 + 2\n"
        assert self.g.grade(code) < 0.7


# ---------------------------------------------------------------------------
# CodeStyleGrader
# ---------------------------------------------------------------------------

class TestCodeStyleGrader:
    def setup_method(self):
        self.g = CodeStyleGrader()

    def test_simple_clean_code(self):
        code = "def add(a, b):\n    return a + b\n"
        assert self.g.grade(code) >= 0.9

    def test_high_complexity_penalised(self):
        # Build a function with many if-else branches
        body = "\n    ".join(f"if x == {i}: return {i}" for i in range(20))
        code = f"def f(x):\n    {body}\n    return -1\n"
        assert self.g.grade(code) < 1.0

    def test_syntax_error_returns_neutral(self):
        assert self.g.grade("def foo(:\n    pass") == 0.5


# ---------------------------------------------------------------------------
# TestResultGrader
# ---------------------------------------------------------------------------

class TestTestResultGrader:
    def setup_method(self):
        self.g = TestResultGrader()

    def test_all_passed(self):
        assert self.g.grade("PASSED: All test cases passed.") == 1.0

    def test_partial_pass(self):
        score = self.g.grade("PASSED: 2/3 test cases passed.")
        assert abs(score - 2/3) < 0.01

    def test_failed_zero(self):
        assert self.g.grade("FAILED:\nAssertionError") == 0.0

    def test_failed_informative(self):
        score = self.g.grade("FAILED:\nAssertionError at line 5: expected 3 got 4")
        assert score == 0.1

    def test_unknown_format_neutral(self):
        assert self.g.grade("some other text") == 0.5


# ---------------------------------------------------------------------------
# ErrorExplanationGrader
# ---------------------------------------------------------------------------

class TestErrorExplanationGrader:
    def setup_method(self):
        self.g = ErrorExplanationGrader()

    def test_specific_explanation_scores_high(self):
        result = (
            "The TypeError at line 3 occurs because you passed a string instead of int. "
            "Try converting with int() before the operation."
        )
        assert self.g.grade(result) >= 0.7

    def test_generic_short_response_scores_low(self):
        assert self.g.grade("Error occurred.") <= 0.2

    def test_empty_scores_low(self):
        assert self.g.grade("ok") <= 0.2


# ---------------------------------------------------------------------------
# SearchRelevanceGrader
# ---------------------------------------------------------------------------

class TestSearchRelevanceGrader:
    def setup_method(self):
        self.g = SearchRelevanceGrader()

    def test_empty_result_zero(self):
        assert self.g.grade("") == 0.0

    def test_no_results_zero(self):
        assert self.g.grade("No results found.") == 0.0

    def test_three_results_good_relevance(self):
        result = "1. def sort_list(lst): ...\n2. def bubble_sort(lst): ...\n3. def merge_sort(lst): ..."
        score = self.g.grade(result, query="sort list python")
        assert score >= 0.5

    def test_single_result_partial(self):
        score = self.g.grade("1. Some example code here that is longer than fifty chars total.")
        assert 0.3 <= score <= 0.7


# ---------------------------------------------------------------------------
# DocQualityGrader
# ---------------------------------------------------------------------------

class TestDocQualityGrader:
    def setup_method(self):
        self.g = DocQualityGrader()

    def test_empty_zero(self):
        assert self.g.grade("") == 0.0

    def test_short_zero(self):
        assert self.g.grade("sorted(...)") == 0.0

    def test_full_docs_score_high(self):
        doc = (
            "sorted(iterable, /, *, key=None, reverse=False)\n"
            "Return a new list containing all items from the iterable in ascending order.\n"
            "A custom key function can be supplied to customize the sort order.\n"
            "Example:\n    >>> sorted([3, 1, 2])\n    [1, 2, 3]\n"
        )
        assert self.g.grade(doc) >= 0.5

    def test_truncated_penalised(self):
        base = "def foo(x):\n    'returns x squared'\n    return x ** 2\n" * 10
        score_truncated = self.g.grade(base + "...")
        score_clean = self.g.grade(base)
        assert score_truncated <= score_clean


# ---------------------------------------------------------------------------
# ReasoningGrader
# ---------------------------------------------------------------------------

class TestReasoningGrader:
    def setup_method(self):
        self.g = ReasoningGrader()

    def test_empty_zero(self):
        assert self.g.grade("") == 0.0

    def test_very_short_low(self):
        assert self.g.grade("ok") <= 0.2

    def test_dense_technical_text_scores_higher(self):
        text = (
            "The BM25 retrieval algorithm uses inverse document frequency weighting "
            "combined with term frequency saturation to rank documents by relevance."
        )
        assert self.g.grade(text) >= 0.5

    def test_filler_text_scores_low(self):
        text = "the is a and or the to of in it is the and but not"
        assert self.g.grade(text) < 0.3


# ---------------------------------------------------------------------------
# RedundancyGrader
# ---------------------------------------------------------------------------

class TestRedundancyGrader:
    def setup_method(self):
        self.g = RedundancyGrader()

    def test_no_redundancy(self):
        log = [
            {"tool": "run_tests", "args": {"code": "x"}, "result": "PASSED"},
            {"tool": "lint_code", "args": {"code": "x"}, "result": "No issues"},
        ]
        penalties = self.g.grade_log(log)
        assert all(p == 0.0 for p in penalties)

    def test_exact_duplicate_penalised(self):
        entry = {"tool": "run_tests", "args": {"code": "x"}, "result": "PASSED"}
        log = [entry, dict(entry)]
        penalties = self.g.grade_log(log)
        assert penalties[1] < 0.0

    def test_duplicate_outside_window_not_penalised(self):
        entry = {"tool": "run_tests", "args": {"code": "x"}, "result": "PASSED"}
        filler = [{"tool": "lint_code", "args": {"code": str(i)}, "result": "ok"} for i in range(5)]
        log = [entry] + filler + [dict(entry)]
        penalties = self.g.grade_log(log)
        assert penalties[-1] == 0.0


# ---------------------------------------------------------------------------
# ErrorPropagationGrader
# ---------------------------------------------------------------------------

class TestErrorPropagationGrader:
    def setup_method(self):
        self.g = ErrorPropagationGrader()

    def test_error_then_explain_gets_bonus(self):
        log = [
            {"tool": "run_tests", "args": {"code": "x"}, "result": "", "error": "NameError"},
            {"tool": "explain_error", "args": {"traceback_text": "NameError"}, "result": "fix it"},
        ]
        adj = self.g.grade_log(log)
        assert adj[1] > 0.0

    def test_error_then_same_args_penalised(self):
        log = [
            {"tool": "run_tests", "args": {"code": "bad_code"}, "result": "", "error": "SyntaxError"},
            {"tool": "run_tests", "args": {"code": "bad_code"}, "result": "", "error": "SyntaxError"},
        ]
        adj = self.g.grade_log(log)
        assert adj[1] < 0.0

    def test_no_error_no_adjustment(self):
        log = [
            {"tool": "run_tests", "args": {"code": "x"}, "result": "PASSED", "error": None},
            {"tool": "lint_code", "args": {"code": "x"}, "result": "ok", "error": None},
        ]
        adj = self.g.grade_log(log)
        assert all(a == 0.0 for a in adj)


# ---------------------------------------------------------------------------
# GraderRouter — integration
# ---------------------------------------------------------------------------

class TestGraderRouter:
    def setup_method(self):
        self.router = GraderRouter()

    def test_grade_test_result_entry(self):
        entry = {"tool": "run_tests", "args": {}, "result": "PASSED: All test cases passed.", "error": None}
        score = self.router.grade(entry)
        assert score == 1.0

    def test_grade_errored_entry_capped(self):
        entry = {"tool": "run_tests", "args": {}, "result": "", "error": "Traceback...", "graded": False}
        score = self.router.grade(entry)
        assert score <= 0.2

    def test_skip_already_graded(self):
        entry = {"tool": "run_tests", "args": {}, "result": "FAILED", "graded": True, "grade": 0.99}
        scores = self.router.grade_log([entry])
        assert scores[0] == 0.99

    def test_grade_log_sets_graded_flag(self):
        entry = {"tool": "run_tests", "args": {}, "result": "PASSED: All test cases passed.", "error": None}
        self.router.grade_log([entry])
        assert entry["graded"] is True
        assert entry["grade"] is not None

    def test_grade_log_multiple_types(self):
        log = [
            {"tool": "search_code_examples", "args": {"query": "sort list"}, "result": "1. def sort(lst): ...\n2. def bubble_sort(lst): ...\n3. extra result here for count", "error": None},
            {"tool": "run_tests", "args": {}, "result": "PASSED: All test cases passed.", "error": None},
            {"tool": "lint_code", "args": {}, "result": "def add(a, b):\n    return a + b\n", "error": None},
        ]
        scores = self.router.grade_log(log)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Test result should score 1.0
        assert scores[1] == 1.0

    def test_redundancy_applied_in_grade_log(self):
        entry = {"tool": "run_tests", "args": {"code": "x"}, "result": "PASSED: All test cases passed.", "error": None}
        log = [entry, dict(entry)]
        scores = self.router.grade_log(log)
        # Second entry should be penalised relative to first
        assert scores[1] < scores[0]
