"""search_code_examples — naive BM25 keyword search over a small built-in corpus.

Initial quality: MEDIUM. Improvable: only lexical matching, no semantic similarity,
no context filtering, corpus is tiny and static.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `search_code_examples(query: str) -> str` must be preserved.
"""

from rank_bm25 import BM25Okapi

_CORPUS = [
    {
        "title": "most_frequent with Counter",
        "code": "from collections import Counter\ndef most_frequent(lst):\n    return Counter(lst).most_common(1)[0][0]",
        "tags": ["counter", "frequency", "list"],
    },
    {
        "title": "binary search",
        "code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1",
        "tags": ["search", "binary", "sorted"],
    },
    {
        "title": "linked list cycle detection",
        "code": "def has_cycle(head):\n    slow, fast = head, head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow is fast: return True\n    return False",
        "tags": ["linked list", "cycle", "floyd"],
    },
    {
        "title": "fibonacci iterative",
        "code": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
        "tags": ["fibonacci", "iterative", "sequence"],
    },
    {
        "title": "two sum hash map",
        "code": "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen: return [seen[target - n], i]\n        seen[n] = i",
        "tags": ["array", "hash", "two sum"],
    },
    {
        "title": "flatten nested list",
        "code": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result",
        "tags": ["flatten", "recursive", "list"],
    },
    {
        "title": "merge sorted arrays",
        "code": "def merge(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: result.append(a[i]); i += 1\n        else: result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]",
        "tags": ["merge", "sort", "array"],
    },
    {
        "title": "palindrome check",
        "code": "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]",
        "tags": ["palindrome", "string"],
    },
    {
        "title": "missing number XOR",
        "code": "def missing_number(nums):\n    return sum(range(len(nums) + 1)) - sum(nums)",
        "tags": ["missing", "xor", "array", "math"],
    },
    {
        "title": "single number XOR",
        "code": "def single_number(nums):\n    result = 0\n    for n in nums: result ^= n\n    return result",
        "tags": ["xor", "single", "duplicate"],
    },
]

_tokenized = [
    (item["title"] + " " + " ".join(item["tags"])).lower().split()
    for item in _CORPUS
]
_bm25 = BM25Okapi(_tokenized)


def search_code_examples(query: str) -> str:
    """Search a local code corpus for examples matching the query.

    Uses BM25 keyword search — returns top-3 results as formatted text.
    """
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            item = _CORPUS[idx]
            results.append(f"### {item['title']}\n```python\n{item['code']}\n```")

    if not results:
        return f"No code examples found for query: {query}"

    return "\n\n".join(results)
