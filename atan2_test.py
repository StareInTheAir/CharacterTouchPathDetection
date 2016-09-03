import math


def main():
    print(atan2_test(0, 0, 0, 0))

    print()

    print(atan2_test(0, 0, 1, 0))  # >
    print(atan2_test(0, 0, 0, 1))  # ^
    print(atan2_test(0, 0, 1, 1))  # ^>

    print()

    print(atan2_test(0, 0, -1, 0))  # <
    print(atan2_test(0, 0, 0, -1))  # v
    print(atan2_test(0, 0, -1, -1))  # v<

    print()

    print(atan2_test(0, 0, 1, -1))  # v>
    print(atan2_test(0, 0, -1, 1))  # ^<

    print()

    print(atan2_test(0, 0, -1, 10.000000000))


def atan2_test(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))
