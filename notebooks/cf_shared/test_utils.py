import sys
sys.path.append("..")
from cf_shared.utils import filter_max_filesize

def test_filter_max_filesize():
    paths = ["one", "two", "three"]
    filesizes = [2, 30, 10]

    assert filter_max_filesize(paths, filesizes, 2) == []
    assert filter_max_filesize(paths, filesizes, 10) == ["one"]
    assert filter_max_filesize(paths, filesizes, 11) == ["one", "three"]
    assert filter_max_filesize(paths, filesizes, 1000) == ["one", "two", "three"]