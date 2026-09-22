"""Validate the model card's x-first, line-per-element grounding format."""
from __future__ import annotations

import re

ELEMENT = re.compile(r'^(p|o|t|pe|ac|fc)(?::([0-9]+))?\[(\d+),(\d+),(\d+),(\d+)\](.*)$')


def validate_grounding(prompt: str) -> None:
    panels = []
    seen_contents = False
    for line in prompt.splitlines():
        if not re.match(r'^(?:p|o|t|pe|ac|fc)(?::|\[)', line):
            continue
        match = ELEMENT.fullmatch(line)
        if not match:
            raise ValueError(f'Malformed grounding element: {line}')
        tag, identity, *rest = match.groups()
        x0, y0, x1, y1 = map(int, rest[:4])
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError(f'Invalid x-first box: {line}')
        if identity is not None and tag not in ('pe', 'ac', 'fc'):
            raise ValueError(f'Only characters take an ID: {line}')
        if tag == 'p':
            if seen_contents:
                raise ValueError('Panels must precede their contents')
            panels.append((x0, y0, x1, y1))
        else:
            seen_contents = True
            if panels and not any(a <= x0 < x1 <= c and b <= y0 < y1 <= d
                                  for a, b, c, d in panels):
                raise ValueError(f'Element outside its panel: {line}')
        if tag == 't' and not re.match(r'^"(?:[^"\\]|\\.)*"', rest[4]):
            raise ValueError(f'Text needs a quoted literal directly after its box: {line}')


def check_token_budget(tokenizer, prompt: str, prefix: str, prefix_tokens: int = 34) -> int:
    """Count using the actual Krea tokenizer and template; reject truncation."""
    count = len(tokenizer(prefix + prompt, add_special_tokens=True)['input_ids']) - prefix_tokens
    if count > 507:
        raise ValueError(f'Prompt uses {count} tokens; Krea permits 507 before its suffix')
    return count
