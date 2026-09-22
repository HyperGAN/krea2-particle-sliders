from pathlib import Path

import pytest

from krea2.grounding import validate_grounding, check_token_budget, ELEMENT
from krea2.prompts import load_prompts


@pytest.mark.parametrize('concept', ['final-boss', 'eldritch'])
def test_concept_pairs_preserve_layout(concept):
    rows, meta = load_prompts(Path(__file__).resolve().parents[1] / f'configs/krea2/prompts-{concept}.yaml')
    assert meta.bare_captions
    assert len(rows) == 6
    for row in rows:
        validate_grounding(row.neutral)
        validate_grounding(row.positive)
        assert row.target == row.neutral
        assert row.positive != row.neutral
        def layout(text):
            return [ELEMENT.fullmatch(line).groups()[:6] for line in text.splitlines()
                    if ELEMENT.fullmatch(line)]
        assert layout(row.neutral) == layout(row.positive)
    assert 'p[500,0,1000,1000]' in rows[-1].positive
    assert rows[-1].positive.count('ac:1[') == 2


@pytest.mark.parametrize('prompt', [
    'o[300,0,200,900] backwards',
    'pe:knight[0,0,1000,1000] unsupported named ID',
    'o[0,0,1001,1000] off canvas',
    't[0,0,500,100] unquoted text',
    'p[0,0,500,1000] left\nt[510,30,800,80]"outside"',
    'o[0,0,500,500] object\np[0,0,1000,1000] late panel',
])
def test_reject_invalid_grounding(prompt):
    with pytest.raises(ValueError):
        validate_grounding(prompt)


def test_budget_includes_template_and_reserves_suffix():
    def tokenizer(text, **kwargs):
        return {'input_ids': list(range(len(text)))}
    assert check_token_budget(tokenizer, 'x' * 507, 'p' * 34) == 507
    with pytest.raises(ValueError, match='508'):
        check_token_budget(tokenizer, 'x' * 508, 'p' * 34)
