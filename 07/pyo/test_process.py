import process

def test_example():
    one, two = process.read_file("../example.txt")
    assert one == 3749
    assert two == 11387
