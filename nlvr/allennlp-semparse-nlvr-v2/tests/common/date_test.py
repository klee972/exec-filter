import pytest

from .. import SemparseTestCase

from allennlp_semparse.common import Date, ExecutionError


class TestDate(SemparseTestCase):
    def test_date_comparison_works(self):
        assert Date(2013, 12, 31) > Date(2013, 12, 30)
        assert Date(2013, 12, 31) == Date(2013, 12, -1)
        assert Date(2013, -1, -1) >= Date(2013, 12, 31)
        assert (Date(2013, 12, -1) > Date(2013, 12, 31)) is False
        with pytest.raises(ExecutionError, match="only compare Dates with Dates"):
            assert (Date(2013, 12, 31) > 2013) is False
        with pytest.raises(ExecutionError, match="only compare Dates with Dates"):
            assert (Date(2013, 12, 31) >= 2013) is False
        with pytest.raises(ExecutionError, match="only compare Dates with Dates"):
            assert Date(2013, 12, 31) != 2013
        assert (Date(2018, 1, 1) >= Date(-1, 2, 1)) is False
        assert (Date(2018, 1, 1) < Date(-1, 2, 1)) is False
        # When year is unknown in both cases, we can compare months and days.
        assert Date(-1, 2, 1) < Date(-1, 2, 3)
        # If both year and month are not know in both cases, the comparison is undefined, and both
        # < and >= return False.
        assert (Date(-1, -1, 1) < Date(-1, -1, 3)) is False
        assert (Date(-1, -1, 1) >= Date(-1, -1, 3)) is False
        # Same when year is known, but months are not.
        assert (Date(2018, -1, 1) < Date(2018, -1, 3)) is False
        assert (Date(2018, -1, 1) >= Date(2018, -1, 3)) is False
