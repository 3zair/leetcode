# 855
import re
from collections import defaultdict, Counter
from itertools import count
from typing import List


class ExamRoom:

    def __init__(self, n: int):
        self.count = 0
        self.n = n
        self.location = [0] * n
        return

    def seat(self) -> int:
        ret = 0
        if self.count == 0:
            ret = 0
        elif self.count == 1 and self.location[0] == 1:
            ret = self.n - 1
        else:
            # left len flag
            max_blank = [-1, 0, 0]
            i = 0
            blank = [-1, 0, 0]
            while i < self.n:
                if self.location[i] == 1:
                    if blank[1] > max_blank[1]:
                        max_blank = blank
                        if max_blank[1] % 2 != 0:
                            max_blank[2] = 1
                            max_blank[1] += 1
                    blank = [i, 0, 0]
                else:
                    blank[1] += 1
                i += 1
            if blank[1] > max_blank[1] or (max_blank[0] == -1 and blank[1] > max_blank[1] - max_blank[2]):
                ret = self.n - 1
            elif max_blank[0] == -1:
                ret = 0
            else:
                ret = max_blank[0] + int(max_blank[1] / 2)

        self.count += 1
        self.location[ret] = 1
        return ret

    def leave(self, p: int) -> None:
        self.location[p] = 0
        self.count -= 1
        return None


#
def minMovesToSeat(seats: List[int], students: List[int]) -> int:
    seats.sort()
    students.sort()
    # for i in range(0, len(seats)):
    #     ret += abs(seats[i] - students[i])
    return sum(abs(x - y) for x, y in zip(seats, students))


# 2351
def repeatedCharacter(s: str) -> str:
    tem = defaultdict(int)
    for c in s:
        if tem[c] == 1:
            return c
        tem[c] = 1
    return ""


#
class SellOrder(object):
    def __init__(self, price, count):
        self.price = price
        self.count = count

    def __lt__(self, other):
        return self.price < other.price

    def __str__(self):
        return "price:{}, count:{}".format(self.price, self.count)


#
class BuyOrder(object):
    def __init__(self, price, count):
        self.price = price
        self.count = count

    def __lt__(self, other):
        return self.price > other.price

    def __str__(self):
        return "price:{}, count:{}".format(self.price, self.count)


from queue import PriorityQueue


def getNumberOfBacklogOrders(orders: List[List[int]]) -> int:
    mod = 10 ** 9 + 7
    sell_queue = PriorityQueue()
    buy_queue = PriorityQueue()
    for order in orders:
        while order[1] > 0:
            if order[2] == 1:
                if buy_queue.empty():
                    sell_queue.put(SellOrder(order[0], order[1]))
                    break
                buyO = buy_queue.queue[0]
                if buyO.price >= order[0]:
                    if order[1] <= buyO.count:
                        buyO.count = buyO.count - order[1]
                        buy_queue.get()
                        buy_queue.put(buyO)
                        break
                    else:
                        order[1] -= buyO.count
                        buy_queue.get()
                else:
                    sell_queue.put(SellOrder(order[0], order[1]))
                    break
            else:
                if sell_queue.empty():
                    buy_queue.put(BuyOrder(order[0], order[1]))
                    break
                sellO = sell_queue.queue[0]
                if sellO.price <= order[0]:
                    if order[1] <= sellO.count:
                        sellO.count = sellO.count - order[1]
                        sell_queue.get()
                        sell_queue.put(sellO)
                        break
                    else:
                        order[1] -= sellO.count
                        sell_queue.get()
                else:
                    buy_queue.put(BuyOrder(order[0], order[1]))
                    break
    ret = 0


def minOperations(nums: List[int], x: int) -> int:
    l = -1
    r = len(nums)
    while l < r:
        if (sum(nums[l:0]) + sum(nums[r:len(nums) - 1])) > x:
            print()

    return 0


def reinitializePermutation(n: int) -> int:
    res = 0
    perm = [i for i in range(0, n)]
    while True:
        mid = n // 2 - 1
        la = perm[0:mid + 1]
        ra = perm[mid + 1:n]
        arr = []
        i = 0
        j = 0
        flag = False
        while i <= mid:
            if la[i] ^ j > 0:
                flag = True
            arr.append(la[i])
            j += 1
            if ra[i] ^ j > 0:
                flag = True
            arr.append(ra[i])
            i += 1
            j += 1
        res += 1
        if flag:
            perm = arr
        else:
            return res


def digitCount(num: str) -> bool:
    count = [0] * len(num)

    for ch in num:
        ch = ord(ch) - ord('0')
        if ch >= len(num):
            return False
        count[ch] += 1
    for i, ch in enumerate(num):
        if count[i] != ord(ch) - ord('0'):
            return False
    return True


def evaluate(s: str, knowledge: List[List[str]]) -> str:
    # k_dict = {}
    # for k in knowledge:
    #     k_dict[k[0]] = k[1]
    # res = ""
    # i = 0
    # while i < len(s):
    #     if s[i] == '(':
    #         i += 1
    #         key = ""
    #         while s[i] != ")":
    #             key += s[i]
    #             i += 1
    #         if key in knowledge:
    #             i += 1
    #         if key in k_dict:
    #             res += k_dict[key]
    #         else:
    #             res += "?"
    #     else:
    #         res += s[i]
    #     print(i,s[i])
    #     i += 1
    for k in knowledge:
        s = s.replace("({})".format(k[0]), k[1])
    return re.sub(r"\([a-z]*\)", "?", s)


def rearrangeCharacters(s: str, target: str) -> int:
    target_c = Counter(target)
    s_c = Counter(s)
    res = 1e8
    for ch, count in target_c.items():
        res = min(res, s_c[ch] // count)
    if res == 1e8:
        return 0
    return res


def minMaxGame(nums: List[int]) -> int:
    new_nums = []
    while True:
        n = len(nums)
        if n == 1:
            return nums[0]
        for i in range(n // 2):
            if i % 2 == 0:
                new_nums.append(min(nums[i * 2], nums[i * 2 + 1]))
            else:
                new_nums.append(max(nums[i * 2], nums[i * 2 + 1]))

        nums = new_nums
        new_nums = []


def areSentencesSimilar(sentence1: str, sentence2: str) -> bool:
    max_sen, min_sen = (sentence1, sentence2) if len(sentence1) > len(sentence2) else (sentence2, sentence1)
    sen1 = max_sen.split(" ")
    sen2 = min_sen.split(" ")
    i = 0

    return 0


def rev(n):
    rev = 0
    while n > 0:
        rev *= 10
        rev += n % 10
        n //= 10

    return rev


from scipy.special import comb


def countNicePairs(nums: List[int]) -> int:
    diff = {}
    mod = 1e9 + 7
    res = 0
    for num in nums:
        d = num - rev(num)
        if d not in diff:
            diff[d] = 0
        diff[d] += 1

    for k, count in diff.items():
        # print(k, count)
        res += comb(count, 2)
        res %= mod
    return int(res)


def strongPasswordCheckerII(password: str) -> bool:
    if len(password) < 8:
        return False
    last = ""
    flags = [False] * 4
    special = "!@#$%^&*()-+"
    for ch in password:
        if last == ch:
            return False
        if ord('a') < ord(ch) < ord('z'):
            flags[0] = True
        elif ord('A') < ord(ch) < ord('Z'):
            flags[1] = True
        elif ord('0') < ord(ch) < ord('9'):
            flags[2] = True
        elif ch in special:
            flags[3] = True
        last = ch
        if flags[0] and flags[1] and flags[2] and flags[3]:
            return True
    return False


def minSideJumps(obstacles: List[int]) -> int:
    cur = 2
    ans = 0
    i = 0
    while i < len(obstacles):
        while obstacles[i] != cur:
            i += 1
            if i == len(obstacles):
                return ans
        ans += 1
        i += 1
        pre = obstacles[i]
        while i < len(obstacles) and (obstacles[i] == 0 or obstacles[i] == cur or obstacles[i] == pre):
            if pre == 0:
                pre = obstacles[i]
                i += 1
            else:
                cur = obstacles[i]
                break
        i += 1
    return ans


def getSmallestString(n: int, k: int) -> str:
    base = ord('a') - 1
    s = []
    for i in range(1, n + 1):
        num = max(1, k - (n - i) * 26)
        k -= num
        s.append(chr(base + num))
    return ''.join(s)


def greatestLetter(s: str) -> str:
    ans = ''.join(set(s))
    ans = list(ans.upper())
    ans.sort(reverse=True)
    for i in range(1, len(ans)):
        if ans[i] == ans[i - 1]:
            return ans[i]
    return ""


import numpy as np


def waysToMakeFair(nums: List[int]) -> int:
    ans = 0
    for i in range(len(nums)):
        nn = nums.copy()
        del nn[i]
        nn = np.array(nn)
        nn[::2] *= -1
        if sum(nn) == 0:
            ans += 1
    return ans


def countAsterisks(s: str) -> int:
    ans = 0
    s_l = s.split("|")
    for i in range(len(s_l)):
        if i % 2 == 0:
            ans += s_l[i].count("*")
    return ans


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
    head = ListNode
    head.next = list1
    pre = head
    cur = list1
    index = 0
    while index != b:
        if index == a:
            pre.next = list2
        pre = cur
        cur = cur.next
        index += 1
    l2 = list2
    while l2.next is not None:
        l2 = l2.next

    l2.next = cur.next
    return head.next


if __name__ == '__main__':
    res = countAsterisks('l|*e*et|c**o|*de|')
    # res = waysToMakeFair([1, 2, 3])
    # er = ExamRoom(8)
    # print(er.seat())
    # print(er.seat())
    # ret = minMovesToSeat([4, 1, 5, 9], [1, 3, 2, 6])
    # print(ret)
    # mask = defaultdict(int)
    # print(mask)
    # print(mask[0])
    # print(mask[2])
    # ret = repeatedCharacter("abccbaacz")
    # ret = getNumberOfBacklogOrders([[10, 5, 0], [15, 2, 1], [25, 1, 1], [30, 4, 0]])
    # ret = minOperations([1, 1, 4, 2, 3], 5)
    # res = reinitializePermutation(1000)
    # res = digitCount("030")
    # res = evaluate("(name)is(age)yearsold", [["name", "bob"], ["age", "two"]])
    # res = rearrangeCharacters("abcba", "abc")
    # res = minMaxGame([1, 3, 5, 2, 4, 8, 2, 2])
    # res = areSentencesSimilar("xD iP tqchblXgqvNVdi", "FmtdCzv Gp YZf UYJ xD iP tqchblXgqvNVdi")
    # res = countNicePairs([13, 10, 35, 24, 76])
    # print(not 7 ^ 7)
    # res = minSideJumps([0, 2, 1, 0, 3, 0])
    # res = getSmallestString(5, 73)
    # res = greatestLetter('AbCdEfGhIjK')
    print(res)
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots()
    # # 以水平轴按照 angles 参数逆时针旋转得到箭头方向， units='xy' 指出了箭头长度计算方法
    # ax.quiver((0, 0), (0, 0), (1, 0), (1, 3), angles='xy', units='xy', scale=1, color='r')
    # plt.axis('equal')
    # plt.xticks(range(-5, 6))
    # plt.yticks(range(-5, 6))
    #
    # plt.grid()
    # plt.show()
