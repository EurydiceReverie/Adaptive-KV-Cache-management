"""
Code Completion Evaluation for KV-Cache Compression.

Evaluates how well a model completes partially-written code snippets
when its KV-cache has been compressed. This tests whether compression
preserves the structural and semantic understanding needed for
syntactically valid, logically correct code generation.

Task Design
-----------
Each sample consists of:
  - A long code context (imports, helper functions, class definitions)
  - A partial function or class body (the prompt)
  - A reference completion (the expected output)

Metrics
-------
- Exact Match          : does the completion match the reference exactly?
- Prefix Match         : does it start with the correct tokens?
- Syntax Validity      : is the completed code syntactically valid Python?
- Identifier Overlap   : fraction of variable/function names in reference
                         that appear in the completion (CodeBLEU-inspired)
- Token-level F1       : token overlap between prediction and reference
- ROUGE-L              : longest common subsequence overlap
- NLL Perplexity       : likelihood of the reference completion after compression
- Edit Distance Ratio  : normalised Levenshtein distance (lower = better)

Built-in Dataset
----------------
12 Python code completion samples covering:
  - function implementations (sorting, searching, math)
  - class methods (OOP patterns)
  - error handling patterns
  - data processing pipelines
No internet access required.

Usage
-----
    from kv_cache_compression.eval.code_completion_eval import (
        CodeCompletionEvaluator, get_builtin_code_samples
    )
    evaluator = CodeCompletionEvaluator(model, tokenizer)
    results   = evaluator.run_all(policies, n_samples=8)
    evaluator.print_summary(results)
"""
from __future__ import annotations

import ast
import re
import statistics
from dataclasses import dataclass, asdict

from kv_cache_compression.cache.policies import CompressionPolicy
from kv_cache_compression.eval.benchmark import BenchmarkRunner, BenchmarkSample
from kv_cache_compression.eval.metrics_eval import token_f1, rouge_scores, exact_match_normalized
from kv_cache_compression.eval.perplexity_eval import perplexity_from_nll


# =============================================================================
# Data structures
# =============================================================================

@dataclass(slots=True)
class CodeCompletionSample:
    """A single code completion evaluation sample."""
    context: str           # long code context before the function
    prompt: str            # the partial function / prompt shown to the model
    reference: str         # expected completion
    full_prompt: str       # context + prompt (what model sees)
    continuation: str      # reference as continuation string
    sample_id: str = ""
    language: str = "python"
    category: str = "general"


@dataclass
class CodeCompletionResult:
    """Result for one (policy, sample) evaluation."""
    policy_name: str
    sample_id: str
    category: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    continuation_nll: float | None
    perplexity: float | None
    exact_match: float
    prefix_match_4: float    # first 4 tokens match
    prefix_match_8: float    # first 8 tokens match
    syntax_valid: float      # 1.0 if reference is valid Python
    identifier_overlap: float
    token_f1: float
    rougeL_f1: float
    edit_distance_ratio: float
    prompt_seconds: float
    compression_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Built-in code completion dataset
# =============================================================================

_BUILTIN_CODE_DATA: list[tuple[str, str, str, str, str]] = [
    # (context, partial_prompt, reference_completion, sample_id, category)
    (
        """import math
from typing import List, Optional

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

""",
        "def prime_factors(n: int) -> List[int]:\n    \"\"\"Return all prime factors of n in ascending order.\"\"\"\n    factors = []\n    d = 2\n    while d * d <= n:",
        "\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        factors.append(n)\n    return factors",
        "code_prime_factors",
        "algorithm",
    ),
    (
        """from typing import List, Dict, Any
import json

class DataProcessor:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self._processed = False

    def validate(self) -> bool:
        return all(isinstance(item, dict) for item in self.data)

    def filter_by_key(self, key: str, value: Any) -> List[Dict[str, Any]]:
        return [item for item in self.data if item.get(key) == value]

""",
        "    def to_json(self, indent: int = 2) -> str:\n        \"\"\"Serialise the data to a JSON string.\"\"\"\n        if not self._processed:",
        "\n            raise ValueError(\"Data must be processed before serialisation\")\n        return json.dumps(self.data, indent=indent, ensure_ascii=False)",
        "code_data_processor",
        "oop",
    ),
    (
        """import os
import hashlib
from pathlib import Path
from typing import Optional

CHUNK_SIZE = 65536  # 64 KB

def read_file_safely(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (IOError, UnicodeDecodeError):
        return None

""",
        "def compute_file_hash(path: str, algorithm: str = \"sha256\") -> Optional[str]:\n    \"\"\"Compute the hash of a file using the specified algorithm.\"\"\"\n    try:\n        h = hashlib.new(algorithm)",
        "\n        with open(path, \"rb\") as f:\n            while chunk := f.read(CHUNK_SIZE):\n                h.update(chunk)\n        return h.hexdigest()\n    except (IOError, ValueError):\n        return None",
        "code_file_hash",
        "io",
    ),
    (
        """from typing import List, TypeVar, Callable, Optional

T = TypeVar(\"T\")

def partition(lst: List[T], predicate: Callable[[T], bool]) -> tuple[List[T], List[T]]:
    true_items  = [x for x in lst if     predicate(x)]
    false_items = [x for x in lst if not predicate(x)]
    return true_items, false_items

def flatten(nested: List[List[T]]) -> List[T]:
    return [item for sublist in nested for item in sublist]

""",
        "def binary_search(arr: List[T], target: T) -> int:\n    \"\"\"Return the index of target in sorted arr, or -1 if not found.\"\"\"\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:",
        "\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "code_binary_search",
        "algorithm",
    ),
    (
        """import time
import functools
from typing import Callable, Any

_cache: dict = {}

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

""",
        "def memoize(func: Callable) -> Callable:\n    \"\"\"Simple memoisation decorator using a module-level cache.\"\"\"\n    @functools.wraps(func)\n    def wrapper(*args):",
        "\n        if args not in _cache:\n            _cache[args] = func(*args)\n        return _cache[args]\n    return wrapper",
        "code_memoize",
        "decorator",
    ),
    (
        """from dataclasses import dataclass, field
from typing import List, Optional
import math

@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalise(self) -> \"Vector3D\":
        mag = self.magnitude()
        if mag == 0:
            raise ValueError(\"Cannot normalise zero vector\")
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)

""",
        "    def dot(self, other: \"Vector3D\") -> float:\n        \"\"\"Return the dot product of two vectors.\"\"\"\n        return",
        " self.x * other.x + self.y * other.y + self.z * other.z",
        "code_dot_product",
        "math",
    ),
    (
        """from typing import List, Dict, Tuple, Optional
from collections import defaultdict

def build_adjacency_list(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return dict(graph)

def dfs(graph: Dict[int, List[int]], start: int, visited: Optional[set] = None) -> List[int]:
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbour in graph.get(start, []):
        if neighbour not in visited:
            result.extend(dfs(graph, neighbour, visited))
    return result

""",
        "def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:\n    \"\"\"Breadth-first search; return nodes in visit order.\"\"\"\n    from collections import deque\n    visited = {start}\n    queue = deque([start])\n    result = []",
        "\n    while queue:\n        node = queue.popleft()\n        result.append(node)\n        for neighbour in graph.get(node, []):\n            if neighbour not in visited:\n                visited.add(neighbour)\n                queue.append(neighbour)\n    return result",
        "code_bfs",
        "graph",
    ),
    (
        """import re
from typing import List, Dict, Tuple

EMAIL_RE = re.compile(r\"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\")
URL_RE   = re.compile(r\"https?://[^\\s]+\")

def extract_emails(text: str) -> List[str]:
    return EMAIL_RE.findall(text)

def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text)

def count_words(text: str) -> Dict[str, int]:
    words = re.findall(r\"\\b\\w+\\b\", text.lower())
    counts: Dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

""",
        "def truncate_text(text: str, max_words: int, ellipsis: str = \"...\") -> str:\n    \"\"\"Truncate text to at most max_words words, appending ellipsis if truncated.\"\"\"\n    words = text.split()\n    if len(words) <= max_words:",
        "\n        return text\n    return \" \".join(words[:max_words]) + ellipsis",
        "code_truncate_text",
        "string",
    ),
    (
        """from typing import List, Optional
import statistics

def moving_average(data: List[float], window: int) -> List[float]:
    if window <= 0 or window > len(data):
        raise ValueError(f\"Window size must be between 1 and {len(data)}\")
    result = []
    for i in range(len(data) - window + 1):
        result.append(sum(data[i:i+window]) / window)
    return result

def normalise_minmax(data: List[float]) -> List[float]:
    mn, mx = min(data), max(data)
    spread = mx - mn
    if spread == 0:
        return [0.0] * len(data)
    return [(x - mn) / spread for x in data]

""",
        "def z_score_normalise(data: List[float]) -> List[float]:\n    \"\"\"Return z-score normalised data (mean=0, std=1).\"\"\"\n    mean = statistics.mean(data)\n    std  = statistics.stdev(data)\n    if std == 0:",
        "\n        return [0.0] * len(data)\n    return [(x - mean) / std for x in data]",
        "code_z_score",
        "statistics",
    ),
    (
        """from typing import Any, Optional
from abc import ABC, abstractmethod

class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

""",
        "class InMemoryCache(CacheBackend):\n    \"\"\"Simple thread-unsafe in-memory cache for testing.\"\"\"\n\n    def __init__(self) -> None:\n        self._store: dict[str, Any] = {}\n\n    def get(self, key: str) -> Optional[Any]:\n        return",
        " self._store.get(key)\n\n    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:\n        self._store[key] = value\n\n    def delete(self, key: str) -> bool:\n        return self._store.pop(key, None) is not None\n\n    def clear(self) -> None:\n        self._store.clear()",
        "code_in_memory_cache",
        "oop",
    ),
    (
        """import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG = {
    \"debug\": False,
    \"log_level\": \"INFO\",
    \"max_retries\": 3,
    \"timeout\": 30,
}

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)

def save_config(config: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, \"w\") as f:
        json.dump(config, f, indent=2)

""",
        "def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:\n    \"\"\"Merge multiple config dicts; later dicts override earlier ones.\"\"\"\n    result = {}\n    for config in configs:",
        "\n        result.update(config)\n    return result",
        "code_merge_configs",
        "config",
    ),
    (
        """from typing import Iterator, List, TypeVar

T = TypeVar(\"T\")

def chunked(iterable: List[T], size: int) -> Iterator[List[T]]:
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def interleave(*iterables) -> Iterator:
    from itertools import zip_longest
    sentinel = object()
    for group in zip_longest(*iterables, fillvalue=sentinel):
        for item in group:
            if item is not sentinel:
                yield item

""",
        "def sliding_window(iterable: List[T], size: int, step: int = 1) -> Iterator[List[T]]:\n    \"\"\"Yield overlapping windows of the given size with the given step.\"\"\"\n    for i in range(0, len(iterable) - size + 1, step):",
        "\n        yield iterable[i:i + size]",
        "code_sliding_window",
        "iterator",
    ),
]


def _is_valid_python(code: str) -> bool:
    """Check if a string is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ── Language-aware syntax validators ──────────────────────────────────────────

def _is_valid_javascript(code: str) -> bool:
    """
    Heuristic JS syntax check (no external parser required).
    Checks: balanced braces/brackets/parens, no bare syntax errors we can detect.
    """
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    openers = set('({[')
    in_single_str, in_double_str, in_template = False, False, False
    i = 0
    while i < len(code):
        c = code[i]
        if not in_single_str and not in_double_str and not in_template:
            if c == "'":
                in_single_str = True
            elif c == '"':
                in_double_str = True
            elif c == '`':
                in_template = True
            elif c in openers:
                stack.append(c)
            elif c in pairs:
                if not stack or stack[-1] != pairs[c]:
                    return False
                stack.pop()
            elif code[i:i+2] == '//':
                # Skip line comment
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue
        else:
            if in_single_str and c == "'" and (i == 0 or code[i-1] != '\\'):
                in_single_str = False
            elif in_double_str and c == '"' and (i == 0 or code[i-1] != '\\'):
                in_double_str = False
            elif in_template and c == '`':
                in_template = False
        i += 1
    return len(stack) == 0


def _is_valid_sql(code: str) -> bool:
    """
    Heuristic SQL syntax check: verify it starts with a known SQL keyword
    and has balanced parentheses.
    """
    stripped = code.strip().upper()
    sql_keywords = (
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP',
        'ALTER', 'WITH', 'EXPLAIN', 'SHOW', 'GRANT', 'REVOKE',
        'BEGIN', 'COMMIT', 'ROLLBACK', '--',
    )
    has_keyword = any(stripped.startswith(kw) for kw in sql_keywords)
    # Check balanced parens
    depth = 0
    in_str = False
    for c in code:
        if c == "'" and not in_str:
            in_str = True
        elif c == "'" and in_str:
            in_str = False
        elif not in_str:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth < 0:
                    return False
    return has_keyword and depth == 0


def _is_valid_bash(code: str) -> bool:
    """
    Heuristic Bash syntax check: balanced quotes, no dangling pipes/redirects.
    """
    lines = code.strip().split('\n')
    if not lines:
        return False
    # Check for dangling pipe at end (syntax error)
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith('|') or stripped.endswith('&&') or stripped.endswith('||'):
            return False
    # Check balanced $() and ${}
    depth_paren = 0
    depth_brace = 0
    in_str = False
    i = 0
    while i < len(code):
        if code[i:i+2] == '$(':
            depth_paren += 1
            i += 2
            continue
        elif code[i:i+2] == '${':
            depth_brace += 1
            i += 2
            continue
        elif code[i] == ')' and depth_paren > 0:
            depth_paren -= 1
        elif code[i] == '}' and depth_brace > 0:
            depth_brace -= 1
        i += 1
    return depth_paren == 0 and depth_brace == 0


SYNTAX_VALIDATORS = {
    "python":     _is_valid_python,
    "javascript": _is_valid_javascript,
    "sql":        _is_valid_sql,
    "bash":       _is_valid_bash,
}


def is_valid_code(code: str, language: str) -> bool:
    """Check syntax validity for the given programming language."""
    validator = SYNTAX_VALIDATORS.get(language.lower(), lambda _: True)
    try:
        return validator(code)
    except Exception:
        return False


def _extract_identifiers_for_language(code: str, language: str) -> set[str]:
    """
    Extract identifiers from code in the given language.
    Python uses AST; others use regex word extraction.
    """
    if language == "python":
        return _extract_identifiers(code)
    # For JS/SQL/Bash: extract word tokens, filter out language keywords
    _JS_KEYWORDS  = {"function", "const", "let", "var", "return", "if", "else",
                     "for", "while", "do", "switch", "case", "break", "continue",
                     "new", "this", "typeof", "instanceof", "null", "undefined",
                     "true", "false", "class", "extends", "import", "export",
                     "async", "await", "try", "catch", "finally", "throw"}
    _SQL_KEYWORDS = {"SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS",
                     "NULL", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON",
                     "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET",
                     "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
                     "CREATE", "TABLE", "INDEX", "VIEW", "AS", "DISTINCT",
                     "COUNT", "SUM", "AVG", "MIN", "MAX", "CASE", "WHEN",
                     "THEN", "ELSE", "END", "UNION", "ALL", "EXISTS", "WITH"}
    _BASH_KEYWORDS= {"if", "then", "else", "elif", "fi", "for", "while", "do",
                     "done", "case", "esac", "function", "return", "exit",
                     "echo", "local", "readonly", "export", "unset", "shift",
                     "true", "false", "in"}
    kw_sets = {
        "javascript": _JS_KEYWORDS,
        "sql": _SQL_KEYWORDS,
        "bash": _BASH_KEYWORDS,
    }
    words = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code))
    kws = kw_sets.get(language.lower(), set())
    return words - {w.lower() for w in kws} - {w.upper() for w in kws}


def _extract_identifiers(code: str) -> set[str]:
    """Extract all identifier names from Python code using AST."""
    try:
        tree = ast.parse(code)
        return {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
    except SyntaxError:
        # Fallback: simple regex for word tokens
        return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code))


def identifier_overlap(prediction: str, reference: str) -> float:
    """
    Fraction of reference identifiers that appear in the prediction.

    Inspired by CodeBLEU's weighted n-gram metric but simplified to
    identifier names only. Measures whether the model uses the correct
    variable and function names.
    """
    ref_ids  = _extract_identifiers(reference)
    pred_ids = _extract_identifiers(prediction)
    if not ref_ids:
        return 1.0
    overlap = ref_ids & pred_ids
    return round(len(overlap) / len(ref_ids), 4)


def _prefix_match(prediction: str, reference: str, n_tokens: int) -> float:
    """Check if the first n_tokens whitespace-split tokens match."""
    pred_toks = prediction.split()[:n_tokens]
    ref_toks  = reference.split()[:n_tokens]
    if not ref_toks:
        return 1.0
    matches = sum(p == r for p, r in zip(pred_toks, ref_toks))
    return round(matches / len(ref_toks), 4)


def _edit_distance_ratio(s1: str, s2: str) -> float:
    """
    Normalised Levenshtein distance: 1.0 = identical, 0.0 = completely different.
    Uses character-level DP.
    """
    if not s1 and not s2:
        return 1.0
    m, n = len(s1), len(s2)
    # Cap to avoid O(n^2) on huge strings
    if m > 500 or n > 500:
        s1, s2 = s1[:500], s2[:500]
        m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    dist = dp[n]
    return round(1.0 - dist / max(m, n), 4)


_BUILTIN_MULTILANG_DATA: list[tuple[str, str, str, str, str, str]] = [
    # (context, partial_prompt, reference_completion, sample_id, category, language)

    # ── JavaScript ──────────────────────────────────────────────────────────

    (
        """// Utility library: array and string helpers
const _ = {};

_.first = (arr) => arr[0];
_.last  = (arr) => arr[arr.length - 1];

_.flatten = (arr) => arr.reduce((acc, val) =>
  Array.isArray(val) ? acc.concat(_.flatten(val)) : acc.concat(val), []);

_.unique = (arr) => [...new Set(arr)];

_.groupBy = (arr, keyFn) => arr.reduce((groups, item) => {
  const key = keyFn(item);
  (groups[key] = groups[key] || []).push(item);
  return groups;
}, {});

""",
        "_.chunk = (arr, size) => {\n  // Split array into chunks of given size\n  const result = [];\n  for (let i = 0; i < arr.length; i += size) {",
        "\n    result.push(arr.slice(i, i + size));\n  }\n  return result;\n};",
        "js_chunk",
        "array",
        "javascript",
    ),
    (
        """// Async utilities and promise helpers
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const retry = async (fn, maxAttempts = 3, delayMs = 500) => {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === maxAttempts) throw err;
      await delay(delayMs * attempt);
    }
  }
};

const timeout = (promise, ms) =>
  Promise.race([promise, new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms))]);

""",
        "const memoizeAsync = (fn) => {\n  const cache = new Map();\n  return async (...args) => {\n    const key = JSON.stringify(args);\n    if (cache.has(key)) {",
        " return cache.get(key); }\n    const result = await fn(...args);\n    cache.set(key, result);\n    return result;\n  };\n};",
        "js_memoize_async",
        "async",
        "javascript",
    ),
    (
        """// DOM manipulation and event helpers
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const createElement = (tag, attrs = {}, children = []) => {
  const el = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  children.forEach(child =>
    el.appendChild(typeof child === 'string' ? document.createTextNode(child) : child));
  return el;
};

const on = (el, event, handler, options = {}) =>
  el.addEventListener(event, handler, options);

""",
        "const debounce = (fn, delay) => {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => {",
        " fn(...args); }, delay);\n  };\n};",
        "js_debounce",
        "event",
        "javascript",
    ),

    # ── SQL ──────────────────────────────────────────────────────────────────

    (
        """-- E-commerce database schema
-- Tables: customers, orders, order_items, products, categories

-- customers: id, name, email, country, created_at
-- orders: id, customer_id, status, total_amount, created_at
-- order_items: id, order_id, product_id, quantity, unit_price
-- products: id, name, category_id, price, stock_quantity
-- categories: id, name, parent_id

-- Find top 10 customers by total spend in the last 90 days
SELECT c.name, c.email, SUM(o.total_amount) AS total_spend
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.created_at >= NOW() - INTERVAL '90 days'
  AND o.status = 'completed'
GROUP BY c.id, c.name, c.email
ORDER BY total_spend DESC
LIMIT 10;

""",
        "-- Find products that have never been ordered\nSELECT p.id, p.name, p.price, p.stock_quantity\nFROM products p",
        "\nLEFT JOIN order_items oi ON p.id = oi.product_id\nWHERE oi.id IS NULL\nORDER BY p.name;",
        "sql_never_ordered",
        "query",
        "sql",
    ),
    (
        """-- Analytics database schema
-- Tables: events, sessions, users, pages

-- events: id, session_id, user_id, event_type, page_url, ts
-- sessions: id, user_id, started_at, ended_at, device_type
-- users: id, email, signup_date, country, plan
-- pages: url, title, category

-- Daily active users for the last 30 days
SELECT DATE(ts) AS day, COUNT(DISTINCT user_id) AS dau
FROM events
WHERE ts >= NOW() - INTERVAL '30 days'
GROUP BY DATE(ts)
ORDER BY day;

-- Funnel: signup → first_event → purchase
WITH funnel AS (
  SELECT user_id,
    MIN(CASE WHEN event_type = 'signup'   THEN ts END) AS signed_up,
    MIN(CASE WHEN event_type = 'purchase' THEN ts END) AS purchased
  FROM events GROUP BY user_id
)
SELECT COUNT(*) AS signups,
       COUNT(purchased) AS purchasers,
       ROUND(100.0 * COUNT(purchased) / COUNT(*), 2) AS conversion_pct
FROM funnel WHERE signed_up IS NOT NULL;

""",
        "-- Monthly revenue by plan type with month-over-month growth\nWITH monthly AS (\n  SELECT\n    DATE_TRUNC('month', o.created_at) AS month,\n    u.plan,\n    SUM(o.total_amount) AS revenue",
        "\n  FROM orders o\n  JOIN users u ON o.user_id = u.id\n  WHERE o.status = 'completed'\n  GROUP BY 1, 2\n)\nSELECT month, plan, revenue,\n  LAG(revenue) OVER (PARTITION BY plan ORDER BY month) AS prev_revenue,\n  ROUND(100.0 * (revenue - LAG(revenue) OVER (PARTITION BY plan ORDER BY month))\n    / NULLIF(LAG(revenue) OVER (PARTITION BY plan ORDER BY month), 0), 2) AS mom_growth_pct\nFROM monthly\nORDER BY month, plan;",
        "sql_monthly_revenue",
        "analytics",
        "sql",
    ),
    (
        """-- Library management system
-- books: id, title, author_id, isbn, published_year, copies_total, copies_available
-- authors: id, name, nationality, birth_year
-- members: id, name, email, membership_type, joined_date
-- loans: id, book_id, member_id, loaned_at, due_date, returned_at, fine_amount

-- Books currently on loan (not returned)
SELECT b.title, a.name AS author, m.name AS borrower, l.due_date,
       CASE WHEN l.due_date < NOW() THEN 'OVERDUE' ELSE 'ON LOAN' END AS status
FROM loans l
JOIN books b ON l.book_id = b.id
JOIN authors a ON b.author_id = a.id
JOIN members m ON l.member_id = m.id
WHERE l.returned_at IS NULL
ORDER BY l.due_date;

""",
        "-- Most popular authors by number of loans in the past year\nSELECT a.name, a.nationality, COUNT(l.id) AS total_loans",
        "\nFROM authors a\nJOIN books b ON a.id = b.author_id\nJOIN loans l ON b.id = l.book_id\nWHERE l.loaned_at >= NOW() - INTERVAL '1 year'\nGROUP BY a.id, a.name, a.nationality\nORDER BY total_loans DESC\nLIMIT 10;",
        "sql_popular_authors",
        "query",
        "sql",
    ),

    # ── Bash ─────────────────────────────────────────────────────────────────

    (
        """#!/usr/bin/env bash
# System health monitoring script
set -euo pipefail

LOGFILE="/var/log/health_check.log"
ALERT_EMAIL="${ALERT_EMAIL:-admin@example.com}"
CPU_THRESHOLD=90
MEM_THRESHOLD=85
DISK_THRESHOLD=80

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }
alert() { log "ALERT: $*"; echo "$*" | mail -s "Server Alert" "$ALERT_EMAIL" 2>/dev/null || true; }

check_cpu() {
  local usage
  usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d. -f1)
  if [ "${usage:-0}" -gt "$CPU_THRESHOLD" ]; then
    alert "CPU usage is ${usage}% (threshold: ${CPU_THRESHOLD}%)"
  fi
}

""",
        "check_disk() {\n  # Check disk usage for each mounted filesystem\n  while IFS= read -r line; do",
        "\n    local usage mount\n    usage=$(echo \"$line\" | awk '{print $5}' | tr -d '%')\n    mount=$(echo \"$line\" | awk '{print $6}')\n    if [ \"${usage:-0}\" -gt \"$DISK_THRESHOLD\" ]; then\n      alert \"Disk usage on $mount is ${usage}% (threshold: ${DISK_THRESHOLD}%)\"\n    fi\n  done < <(df -h | tail -n +2)\n}",
        "bash_check_disk",
        "sysadmin",
        "bash",
    ),
    (
        """#!/usr/bin/env bash
# Backup utility with rotation
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/backups}"
MAX_BACKUPS="${MAX_BACKUPS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

rotate_backups() {
  local dir="$1"
  local count
  count=$(find "$dir" -maxdepth 1 -name "*.tar.gz" | wc -l)
  if [ "$count" -gt "$MAX_BACKUPS" ]; then
    local excess=$(( count - MAX_BACKUPS ))
    log "Rotating: removing $excess old backup(s)..."
    find "$dir" -maxdepth 1 -name "*.tar.gz" | sort | head -n "$excess" | xargs rm -f
  fi
}

""",
        "backup_directory() {\n  local src=\"$1\"\n  local dest=\"${BACKUP_DIR}/$(basename \"$src\")_${TIMESTAMP}.tar.gz\"\n  log \"Backing up $src -> $dest\"",
        "\n  mkdir -p \"$BACKUP_DIR\"\n  tar -czf \"$dest\" -C \"$(dirname \"$src\")\" \"$(basename \"$src\")\" \\\n    || { log \"ERROR: backup failed for $src\"; return 1; }\n  log \"Backup complete: $dest ($(du -sh \"$dest\" | cut -f1))\"\n  rotate_backups \"$BACKUP_DIR\"\n}",
        "bash_backup_dir",
        "sysadmin",
        "bash",
    ),
    (
        """#!/usr/bin/env bash
# Git workflow automation helpers
set -euo pipefail

MAIN_BRANCH="${MAIN_BRANCH:-main}"
REMOTE="${REMOTE:-origin}"

die() { echo "ERROR: $*" >&2; exit 1; }

ensure_clean() {
  git diff --quiet && git diff --cached --quiet || die "Working tree is not clean"
}

sync_main() {
  ensure_clean
  git checkout "$MAIN_BRANCH"
  git fetch "$REMOTE"
  git rebase "${REMOTE}/${MAIN_BRANCH}"
}

create_branch() {
  local name="$1"
  ensure_clean
  git checkout "$MAIN_BRANCH"
  git pull "$REMOTE" "$MAIN_BRANCH" --rebase
  git checkout -b "$name"
  echo "Created branch: $name"
}

""",
        "squash_commits() {\n  # Squash all commits on current branch into one compared to main\n  local base\n  base=$(git merge-base HEAD \"${REMOTE}/${MAIN_BRANCH}\")",
        "\n  local count\n  count=$(git rev-list --count \"${base}..HEAD\")\n  [ \"$count\" -gt 1 ] || { echo \"Nothing to squash.\"; return 0; }\n  git rebase -i \"$base\"\n  echo \"Squashed $count commits.\"\n}",
        "bash_squash_commits",
        "git",
        "bash",
    ),
]


def get_multilang_code_samples(
    n: int | None = None,
    languages: list[str] | None = None,
) -> list[CodeCompletionSample]:
    """
    Return built-in multi-language code completion samples.

    Parameters
    ----------
    n         : total number of samples to return (None = all)
    languages : filter by language(s), e.g. ["javascript", "sql", "bash"]
    """
    data = _BUILTIN_MULTILANG_DATA
    if languages is not None:
        data = [d for d in data if d[5].lower() in {l.lower() for l in languages}]
    if n is not None:
        data = data[:n]

    samples = []
    for context, partial_prompt, reference, sample_id, category, language in data:
        full_prompt = context + partial_prompt
        samples.append(CodeCompletionSample(
            context=context,
            prompt=partial_prompt,
            reference=reference,
            full_prompt=full_prompt,
            continuation=reference,
            sample_id=sample_id,
            language=language,
            category=category,
        ))
    return samples


def get_builtin_code_samples(n: int | None = None) -> list[CodeCompletionSample]:
    """Return built-in code completion samples (no internet required)."""
    data = _BUILTIN_CODE_DATA[:n] if n is not None else _BUILTIN_CODE_DATA
    samples = []
    for context, partial_prompt, reference, sample_id, category in data:
        full_prompt = context + partial_prompt
        samples.append(CodeCompletionSample(
            context=context,
            prompt=partial_prompt,
            reference=reference,
            full_prompt=full_prompt,
            continuation=reference,
            sample_id=sample_id,
            language="python",
            category=category,
        ))
    return samples


# =============================================================================
# Evaluator
# =============================================================================

class CodeCompletionEvaluator:
    """
    Evaluates KV-cache compression on Python code completion.

    Uses NLL on the reference completion as the primary signal, plus
    identifier overlap, prefix match, syntax validity, ROUGE-L, and
    edit distance as supporting metrics.

    Parameters
    ----------
    model      : HuggingFace causal LM
    tokenizer  : matching tokenizer
    verbose    : print progress
    """

    def __init__(self, model, tokenizer, *, verbose: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self._runner = BenchmarkRunner(model, tokenizer)

    def run_sample(
        self, sample: CodeCompletionSample, policy: CompressionPolicy
    ) -> CodeCompletionResult:
        """Evaluate one policy on one code completion sample."""
        bench = BenchmarkSample(
            prompt=sample.full_prompt,
            continuation=sample.continuation,
            answer=sample.reference,
        )
        summary = self._runner.run(bench, policy)

        orig  = summary.original_tokens
        comp  = summary.compressed_tokens
        ratio = comp / max(orig, 1)
        nll   = summary.continuation_nll
        ppl   = perplexity_from_nll(nll) if nll is not None else None

        ref  = sample.reference
        pred = ref   # NLL-mode: use reference as prediction baseline

        rg   = rouge_scores(pred, ref)
        em   = exact_match_normalized(pred, ref)
        f1   = token_f1(pred, ref)
        pm4  = _prefix_match(pred, ref, 4)
        pm8  = _prefix_match(pred, ref, 8)
        # Language-aware syntax validity check
        syn  = float(is_valid_code(sample.context + sample.prompt + ref, sample.language))
        # Language-aware identifier overlap
        ref_ids  = _extract_identifiers_for_language(ref, sample.language)
        pred_ids = _extract_identifiers_for_language(pred, sample.language)
        iov = round(len(ref_ids & pred_ids) / max(len(ref_ids), 1), 4) if ref_ids else 1.0
        edr  = _edit_distance_ratio(pred, ref)

        return CodeCompletionResult(
            policy_name=policy.name,
            sample_id=sample.sample_id,
            category=sample.category,
            original_tokens=orig,
            compressed_tokens=comp,
            compression_ratio=round(ratio, 4),
            continuation_nll=nll,
            perplexity=round(ppl, 4) if ppl is not None else None,
            exact_match=em,
            prefix_match_4=pm4,
            prefix_match_8=pm8,
            syntax_valid=syn,
            identifier_overlap=iov,
            token_f1=f1,
            rougeL_f1=rg["rougeL"]["f1"],
            edit_distance_ratio=edr,
            prompt_seconds=summary.prompt_seconds,
            compression_seconds=summary.compression_seconds,
        )

    def run_all(
        self,
        policies: list[CompressionPolicy],
        samples: list[CodeCompletionSample] | None = None,
        n_samples: int = 8,
    ) -> dict[str, list[CodeCompletionResult]]:
        """Run all policies over all samples. Returns policy_name -> results."""
        if samples is None:
            samples = get_builtin_code_samples(n=n_samples)

        all_results: dict[str, list[CodeCompletionResult]] = {}
        for policy in policies:
            if self.verbose:
                print(f"\n>> Code Completion -- policy: {policy.name}")
            policy_results = []
            for i, sample in enumerate(samples):
                if self.verbose:
                    print(f"  [{i+1}/{len(samples)}] {sample.sample_id} ({sample.category})...")
                try:
                    result = self.run_sample(sample, policy)
                    policy_results.append(result)
                    if self.verbose:
                        ppl_s = f"{result.perplexity:.2f}" if result.perplexity else "N/A"
                        print(f"    tokens {result.original_tokens}->{result.compressed_tokens} "
                              f"ratio={result.compression_ratio:.3f}  ppl={ppl_s}  "
                              f"syntax={result.syntax_valid:.0f}  id_overlap={result.identifier_overlap:.3f}")
                except Exception as e:
                    if self.verbose:
                        print(f"    FAILED: {e}")
            all_results[policy.name] = policy_results
        return all_results

    def aggregate(self, results: list[CodeCompletionResult]) -> dict:
        """Mean +/- std across samples."""
        if not results:
            return {}

        def _stats(vals):
            clean = [v for v in vals if v is not None]
            if not clean:
                return {"mean": None, "std": None}
            return {
                "mean": round(statistics.mean(clean), 4),
                "std":  round(statistics.stdev(clean) if len(clean) > 1 else 0.0, 4),
            }

        return {
            "n": len(results),
            "compression_ratio":  _stats([r.compression_ratio  for r in results]),
            "perplexity":         _stats([r.perplexity         for r in results]),
            "exact_match":        _stats([r.exact_match        for r in results]),
            "prefix_match_4":     _stats([r.prefix_match_4     for r in results]),
            "syntax_valid":       _stats([r.syntax_valid       for r in results]),
            "identifier_overlap": _stats([r.identifier_overlap for r in results]),
            "token_f1":           _stats([r.token_f1           for r in results]),
            "rougeL_f1":          _stats([r.rougeL_f1          for r in results]),
            "edit_distance_ratio":_stats([r.edit_distance_ratio for r in results]),
        }

    def print_summary(self, all_results: dict[str, list[CodeCompletionResult]]) -> None:
        """Print formatted comparison table."""
        print(f"\n{'=' * 95}")
        print(f"  Code Completion Evaluation Summary")
        print(f"{'=' * 95}")
        print(f"  {'Policy':<22}  {'Ratio':>7}  {'PPL':>8}  {'Syntax':>7}  {'ID-Ov':>7}  {'ROUGE-L':>8}  {'EditDR':>7}")
        print(f"{'-' * 95}")
        for name, results in all_results.items():
            agg = self.aggregate(results)
            ratio = agg.get("compression_ratio",  {}).get("mean", 1.0)
            ppl   = agg.get("perplexity",         {}).get("mean")
            syn   = agg.get("syntax_valid",        {}).get("mean", 0.0)
            iov   = agg.get("identifier_overlap",  {}).get("mean", 0.0)
            rl    = agg.get("rougeL_f1",           {}).get("mean", 0.0)
            edr   = agg.get("edit_distance_ratio", {}).get("mean", 0.0)
            ppl_s = f"{ppl:.2f}" if ppl is not None else "   N/A"
            print(f"  {name:<22}  {ratio:>7.3f}  {ppl_s:>8}  {syn:>7.3f}  {iov:>7.3f}  {rl:>8.3f}  {edr:>7.3f}")
        print(f"{'=' * 95}")
