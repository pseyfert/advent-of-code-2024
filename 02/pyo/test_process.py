import process


def test_example():
    one = process.part_one(process.read_file("../example.txt"))
    assert one == 2
