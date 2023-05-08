def rectangleArea(rectangles: list[list[int]]) -> int:
    xs = set()
    for rectangle in rectangles:
        xs.add(rectangle[0])
        xs.add(rectangle[2])
    area = 0
    x_list = list(xs)
    x_list = sorted(x_list)
    # print(x_list)
    for i in range(len(x_list) - 1):
        lines = [(info[1], info[3]) for info in rectangles if info[0] <= x_list[i] and x_list[i + 1] <= info[2]]
        height, b, t = 0, -1, -1
        lines.sort()
        # print(lines)
        for line in lines:
            if line[0] > t:
                height += t - b
                # print(x_list[i + 1], x_list[i], t, b)
                b, t = line
            elif line[1] > t:
                t = line[1]
        height += t - b
        # print(x_list[i + 1], x_list[i], t, b)
        area += ((x_list[i + 1] - x_list[i]) * height) % (1e9 + 7)
    print(area)


if __name__ == '__main__':
    rectangleArea(rectangles=[[0, 0, 2, 2], [1, 0, 2, 3], [1, 0, 3, 1]])
