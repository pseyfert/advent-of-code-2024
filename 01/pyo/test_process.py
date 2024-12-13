import process


def test_example():
    (one, two) = process.read_file("../example.txt")
    assert one == [1, 2, 3, 3, 3, 4]
    assert two == [3, 3, 3, 4, 5, 9]

def test_one():
    assert 11 == process.part_one("../example.txt")

def test_two():
    assert 31 == process.part_two("../example.txt")
