import process


def test_example():
    (one, two) = process.read_file("../example.txt")
    assert one == 1928
    assert two == 2858
