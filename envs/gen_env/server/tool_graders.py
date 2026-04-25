"""Multi-grader system for per-tool-call evaluation.

Routes each tool call log entry to content-type-appropriate graders based on
what the result *contains*, not which tool produced it. This allows new tools
added during self-evolution to be graded without code changes.

Log entry schema (Dict[str, Any]):
    tool:    str        — tool name
    args:    dict       — kwargs passed to the tool
    result:  str        — string output from the tool (empty string if error)
    error:   str|None   — traceback if tool raised, else None
    graded:  bool       — True if already scored in this session (skip)
    grade:   float|None — cached score [0.0, 1.0] if graded=True
"""

import ast
import re
from enum import Enum
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    CODE = "CODE"
    TEST_RESULT = "TEST_RESULT"
    ERROR_TRACEBACK = "ERROR_TRACEBACK"
    SEARCH_RESULT = "SEARCH_RESULT"
    DOCUMENTATION = "DOCUMENTATION"
    TEXT = "TEXT"


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "from", "or", "and", "but", "not", "this", "that",
    "it", "its", "if", "as", "so", "then", "than", "no", "yes", "your",
    "our", "their", "there", "here", "when", "where", "how", "what", "which",
    "who", "also", "can", "use", "you", "we",
})


class ContentTypeDetector:
    """Detects the type of content in a tool result string."""

    def detect(self, result: str) -> ContentType:
        if not result or len(result.strip()) == 0:
            return ContentType.TEXT

        stripped = result.strip()

        # Test results — check first
        if re.match(r"^(PASSED|FAILED)[:\s]", stripped, re.IGNORECASE):
            return ContentType.TEST_RESULT

        # Error tracebacks
        if any(marker in stripped for marker in (
            "Traceback (most recent call last)",
            "Error:", "Exception:", "SyntaxError:",
            "TypeError:", "ValueError:", "KeyError:", "AttributeError:",
            "NameError:", "ImportError:", "IndentationError:",
        )):
            return ContentType.ERROR_TRACEBACK

        # Search results — numbered list or similarity scores
        lines = stripped.splitlines()
        numbered = sum(1 for l in lines if re.match(r"^\d+\.", l.strip()))
        has_score = bool(re.search(r"score[:\s]+\d+\.?\d*|similarity[:\s]+\d+\.?\d*", stripped, re.IGNORECASE))
        if numbered >= 2 or (len(lines) > 2 and has_score):
            return ContentType.SEARCH_RESULT

        # Documentation — pydoc-style: has signature lines
        has_sig = bool(re.search(r"^\s*(class|def)\s+\w+|Help on|DESCRIPTION|FUNCTIONS", stripped, re.MULTILINE))
        has_desc = len(stripped) > 100
        if has_sig and has_desc:
            return ContentType.DOCUMENTATION

        # Code — contains Python code markers
        code_markers = ("def ", "class ", "import ", "return ", "    if ", "    for ", "    while ")
        multiline = len(lines) > 3
        has_indent = any(l.startswith("    ") for l in lines)
        if multiline and has_indent and any(m in stripped for m in code_markers):
            return ContentType.CODE

        # Single-line code snippet
        if any(m in stripped for m in ("def ", "class ", "import ", "lambda ")):
            return ContentType.CODE

        return ContentType.TEXT


# ---------------------------------------------------------------------------
# Individual graders
# ---------------------------------------------------------------------------

class CodeSyntaxGrader:
    """Grades code for syntactic validity and non-triviality."""

    def grade(self, code: str) -> float:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        score = 1.0
        score += self._check_structure(tree)
        score += self._check_placeholders(code, tree)
        score += self._check_nesting(tree)
        return max(0.0, min(1.0, score))

    def _check_structure(self, tree: ast.AST) -> float:
        func_defs = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        if func_defs == 0:
            return -0.4
        return 0.0

    def _check_placeholders(self, code: str, tree: ast.AST) -> float:
        penalty = 0.0
        # Sole-pass body in functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                real_stmts = [s for s in node.body if not isinstance(s, (ast.Pass, ast.Expr))]
                if not real_stmts:
                    penalty -= 0.15
        # Ellipsis expressions
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                if node.value.value is ...:
                    penalty -= 0.15
        # NotImplementedError raises
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                exc = node.exc
                if exc and isinstance(exc, ast.Call):
                    func = exc.func
                    name = getattr(func, "id", getattr(func, "attr", ""))
                    if name == "NotImplementedError":
                        penalty -= 0.15
        # TODO/FIXME comments
        todo_count = len(re.findall(r"#\s*(TODO|FIXME|XXX|HACK)", code, re.IGNORECASE))
        penalty -= 0.05 * todo_count
        return penalty

    def _check_nesting(self, tree: ast.AST) -> float:
        max_depth = _max_nesting_depth(tree)
        if max_depth > 5:
            return -0.15
        if max_depth > 4:
            return -0.1
        return 0.0


class CodeStyleGrader:
    """Grades code for complexity and naming quality."""

    _LOOP_COUNTERS = frozenset({"i", "j", "k", "n", "x", "y", "z", "m"})

    def grade(self, code: str) -> float:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.5  # syntax already caught by CodeSyntaxGrader

        score = 1.0
        cc = self._cyclomatic_complexity(tree)
        lines = [l for l in code.splitlines() if l.strip()]

        if cc > 15:
            score -= 0.2
        elif cc <= 5 and len(lines) > 5:
            score += 0.1

        # Single-char non-counter variable names
        bad_names = self._bad_names(tree)
        score -= 0.05 * len(bad_names)

        return max(0.0, min(1.0, score))

    def _cyclomatic_complexity(self, tree: ast.AST) -> int:
        cc = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.AsyncWith)):
                cc += 1
            elif isinstance(node, ast.BoolOp):
                cc += len(node.values) - 1
        return cc

    def _bad_names(self, tree: ast.AST) -> List[str]:
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if len(node.id) == 1 and node.id not in self._LOOP_COUNTERS:
                    bad.append(node.id)
        return bad


class TestResultGrader:
    """Grades test execution results by pass rate and feedback quality."""

    def grade(self, result: str) -> float:
        stripped = result.strip()

        if stripped.upper().startswith("PASSED"):
            # Try to parse N/M
            m = re.search(r"(\d+)\s*/\s*(\d+)", stripped)
            if m:
                passed, total = int(m.group(1)), int(m.group(2))
                return passed / total if total > 0 else 1.0
            return 1.0

        if stripped.upper().startswith("FAILED"):
            # Partial credit if failure message is informative
            body = stripped[6:].strip()
            informative = len(body) > 20 and bool(re.search(r"line \d+|\bAssertionError\b|expected|got", body))
            return 0.1 if informative else 0.0

        # Unknown format — neutral
        return 0.5


class ErrorExplanationGrader:
    """Grades error traceback explanations for specificity and actionability."""

    _ERROR_TYPES = re.compile(
        r"\b(TypeError|ValueError|KeyError|AttributeError|NameError|"
        r"ImportError|IndexError|RuntimeError|SyntaxError|IndentationError|"
        r"ZeroDivisionError|FileNotFoundError|OverflowError|StopIteration)\b"
    )
    _FIX_KEYWORDS = re.compile(r"\b(try|check|use|replace|add|remove|fix|ensure|call|wrap|handle)\b", re.IGNORECASE)

    def grade(self, result: str, traceback_context: str = "") -> float:  # noqa: ARG002
        if len(result.split()) < 5:
            return 0.1

        score = 0.0

        # Error type referenced
        if self._ERROR_TYPES.search(result):
            score += 0.3

        # Line number referenced
        if re.search(r"line \d+", result, re.IGNORECASE):
            score += 0.2

        # Fix suggested
        if self._FIX_KEYWORDS.search(result):
            score += 0.3

        # Information density
        density = _information_density(result)
        if density > 0.3:
            score += 0.2

        return max(0.0, min(1.0, score))


class SearchRelevanceGrader:
    """Grades search results for non-emptiness and query relevance."""

    def grade(self, result: str, query: str = "") -> float:
        stripped = result.strip()

        if not stripped or len(stripped) < 50:
            return 0.0

        lower = stripped.lower()
        if "no results" in lower or "not found" in lower:
            return 0.0

        # Count distinct results
        lines = stripped.splitlines()
        result_count = sum(1 for l in lines if re.match(r"^\d+\.", l.strip()) or l.strip().startswith("##"))
        if result_count == 0:
            result_count = max(1, len([l for l in lines if len(l.strip()) > 20]) // 3)

        if result_count == 0:
            return 0.1
        if result_count < 2:
            return 0.4

        # With 3+ results, score based on Jaccard relevance if query provided
        if query and result_count >= 3:
            jaccard = _jaccard(query, stripped)
            if jaccard >= 0.1:
                return min(1.0, 0.6 + jaccard)
            # Even with low term overlap, 3 results is still useful
            return 0.6

        return 0.7


class DocQualityGrader:
    """Grades documentation output for completeness and non-truncation."""

    def grade(self, result: str) -> float:
        stripped = result.strip()

        if len(stripped) < 100:
            return 0.0

        has_sig = bool(re.search(r"^\s*(class|def)\s+\w+|Help on|DESCRIPTION|FUNCTIONS|NAME", stripped, re.MULTILINE))
        has_examples = bool(re.search(r"example|>>>\s+\w+|e\.g\.", stripped, re.IGNORECASE))
        truncated = stripped.endswith("...") or "[truncated]" in stripped

        score = 0.4 if has_sig else 0.2
        if has_examples:
            score += 0.4
        elif len(stripped) > 500:
            score += 0.2
        if truncated:
            score -= 0.2

        return max(0.0, min(1.0, score))


class ReasoningGrader:
    """Catch-all grader for plain text outputs."""

    def grade(self, result: str, context: str = "") -> float:
        words = result.split()
        if len(words) < 5:
            return 0.0
        if len(words) < 15:
            return 0.2

        density = _information_density(result)

        score = 0.0
        if density >= 0.4:
            score = 0.8
        elif density >= 0.2:
            score = 0.6
        elif density >= 0.1:
            score = 0.35
        else:
            score = 0.15

        # Bonus if result references identifiers from context
        if context:
            jaccard = _jaccard(context, result)
            score = min(1.0, score + jaccard * 0.3)

        return score


# ---------------------------------------------------------------------------
# Sequence-level graders
# ---------------------------------------------------------------------------

class RedundancyGrader:
    """Penalises duplicate tool calls within a short window."""

    WINDOW = 3
    PENALTY = 0.3

    def grade_log(self, log: List[Dict]) -> List[float]:
        """Returns per-entry penalty (0 or negative) for redundancy."""
        penalties = [0.0] * len(log)
        for i, entry in enumerate(log):
            start = max(0, i - self.WINDOW)
            for j in range(start, i):
                prev = log[j]
                if (prev.get("tool") == entry.get("tool") and
                        prev.get("args") == entry.get("args")):
                    penalties[i] = -self.PENALTY
                    break
        return penalties


class ErrorPropagationGrader:
    """Penalises calls that ignore a preceding tool error; rewards error handling."""

    def grade_log(self, log: List[Dict]) -> List[float]:
        adjustments = [0.0] * len(log)
        for i in range(1, len(log)):
            prev = log[i - 1]
            curr = log[i]
            if prev.get("error") is not None:
                # Check if same resource is reused without handling
                prev_args = str(prev.get("args", {}))
                curr_args = str(curr.get("args", {}))
                # If next tool is explain_error → reward
                if curr.get("tool") == "explain_error":
                    adjustments[i] = 0.2
                elif _args_overlap(prev_args, curr_args):
                    adjustments[i] = -0.4
        return adjustments


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

class GraderRouter:
    """Routes each tool call log entry to the appropriate grader(s)."""

    def __init__(self) -> None:
        self._detector = ContentTypeDetector()
        self._code_syntax = CodeSyntaxGrader()
        self._code_style = CodeStyleGrader()
        self._test_result = TestResultGrader()
        self._error_exp = ErrorExplanationGrader()
        self._search = SearchRelevanceGrader()
        self._doc = DocQualityGrader()
        self._reasoning = ReasoningGrader()
        self._redundancy = RedundancyGrader()
        self._error_prop = ErrorPropagationGrader()

    def grade(self, entry: Dict[str, Any]) -> float:
        """Grade a single entry; returns score in [0.0, 1.0]."""
        result = entry.get("result") or ""
        args = entry.get("args", {})

        content_type = self._detector.detect(result)

        if content_type == ContentType.CODE:
            s = (self._code_syntax.grade(result) + self._code_style.grade(result)) / 2
        elif content_type == ContentType.TEST_RESULT:
            s = self._test_result.grade(result)
        elif content_type == ContentType.ERROR_TRACEBACK:
            tb = str(args.get("traceback_text", ""))
            s = self._error_exp.grade(result, traceback_context=tb)
        elif content_type == ContentType.SEARCH_RESULT:
            query = str(args.get("query", ""))
            s = self._search.grade(result, query=query)
        elif content_type == ContentType.DOCUMENTATION:
            s = self._doc.grade(result)
        else:
            s = self._reasoning.grade(result)

        # Tool errored out entirely → cap at 0.2
        if entry.get("error") is not None:
            s = min(s, 0.2)

        return max(0.0, min(1.0, s))

    def grade_log(self, log: List[Dict[str, Any]]) -> List[float]:
        """Grade all ungraded entries; cache scores in-place. Returns per-entry scores."""
        scores: List[float] = []
        for entry in log:
            if entry.get("graded") and entry.get("grade") is not None:
                scores.append(float(entry["grade"]))
            else:
                s = self.grade(entry)
                entry["graded"] = True
                entry["grade"] = s
                scores.append(s)

        # Apply sequence-level adjustments
        redundancy_adj = self._redundancy.grade_log(log)
        error_prop_adj = self._error_prop.grade_log(log)

        final: List[float] = []
        for i, s in enumerate(scores):
            adjusted = s + redundancy_adj[i] + error_prop_adj[i]
            log[i]["grade"] = max(0.0, min(1.0, adjusted))
            final.append(log[i]["grade"])

        return final


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _information_density(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return len(content) / len(tokens)


def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower())) - _STOPWORDS
    tb = set(re.findall(r"\w+", b.lower())) - _STOPWORDS
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _max_nesting_depth(tree: ast.AST) -> int:
    """Compute maximum nesting depth of compound statements."""
    def _depth(node: ast.AST, current: int) -> int:
        max_d = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With,
                                   ast.AsyncFor, ast.AsyncWith, ast.Try,
                                   ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                max_d = max(max_d, _depth(child, current + 1))
            else:
                max_d = max(max_d, _depth(child, current))
        return max_d
    return _depth(tree, 0)


def _args_overlap(args_a: str, args_b: str) -> bool:
    """Check if two arg repr strings share a significant token (file path, etc.)."""
    tokens_a = set(re.findall(r"[\w./\\-]{4,}", args_a))
    tokens_b = set(re.findall(r"[\w./\\-]{4,}", args_b))
    return bool(tokens_a & tokens_b)
