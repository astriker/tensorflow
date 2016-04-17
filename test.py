


def test_zip():
    for start, end in zip(range(0, 10000, 128), range(128, 10000, 128)):
        print start, end



if __name__ == '__main__':
    test_zip()