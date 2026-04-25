"""explain_error — diagnoses a Python traceback and returns an explanation.

Initial quality: LOW. Returns a generic template that doesn't reference the
specific line, variable, or expression causing the failure.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `explain_error(traceback_text: str, code: str = "") -> str` must be preserved.
"""


def explain_error(traceback_text: str, code: str = "") -> str:
    """Diagnose a Python traceback and return a targeted explanation.

    Args:
        traceback_text: The full Python traceback string.
        code: Optional — the code that produced the traceback.

    Returns:
        A natural-language explanation (currently generic — improvable).
    """
    lines = [l.strip() for l in traceback_text.strip().splitlines() if l.strip()]
    last_line = lines[-1] if lines else traceback_text

    if "NameError" in last_line:
        category = "undefined variable"
    elif "TypeError" in last_line:
        category = "type mismatch"
    elif "IndexError" in last_line:
        category = "index out of range"
    elif "KeyError" in last_line:
        category = "missing dictionary key"
    elif "AttributeError" in last_line:
        category = "attribute not found"
    elif "ValueError" in last_line:
        category = "invalid value"
    elif "ZeroDivisionError" in last_line:
        category = "division by zero"
    else:
        category = "runtime error"

    # Generic — does NOT parse specific line/variable (improvable)
    return (
        f"The error is a {category}. "
        f"Check your code for issues related to {category}. "
        f"Exception message: {last_line}"
    )
