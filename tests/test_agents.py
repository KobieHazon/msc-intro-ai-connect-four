from pathlib import Path

import pytest

from test import readFile, startTest

FIXTURE_DIRECTORY = Path(__file__).parents[1] / "test_yourself"
CASES = tuple(sorted(FIXTURE_DIRECTORY.glob("*.txt")))


@pytest.mark.parametrize("fixture", CASES, ids=lambda path: path.stem)
def test_recovered_agent_fixture(fixture):
    depth, agent_name, board, expected_column = readFile(fixture)

    assert startTest(depth, agent_name, board) == expected_column


def test_all_recovered_fixtures_are_included():
    assert len(CASES) == 14
