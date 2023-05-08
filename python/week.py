import math
from collections import Counter
from typing import List


def maximumCount(nums: List[int]) -> int:
    l = 0
    r = len(nums) - 1
    mid = (l + r) // 2
    while l <= r:
        if nums[mid] < 0:
            l = mid + 1
        elif nums[mid] > 0:
            r = mid - 1
        else:
            l = mid - 1
            r = mid + 1
            while l >= 0 and nums[l] == 0:
                l -= 1
            while r < len(nums) and nums[r] == 0:
                r += 1
            # print(mid, l, r)
            return max(l + 1, len(nums) - r)

        print(mid, l, r)
        mid = (l + r) // 2
    if r == -1 or l == len(nums):
        return len(nums)
    if nums[l] > 0:
        l -= 1
    if nums[r] < 0:
        r += 1
    return max(l + 1, len(nums) - r)


from queue import PriorityQueue


# import heapq
class item(object):
    def __init__(self, x):
        self.score = x

    def __lt__(self, other):
        return self.score > other.score


def maxKelements(nums: List[int], k: int) -> int:
    nums.sort()
    res = 0
    M = nums.pop()
    ext = PriorityQueue()
    while k > 0:
        k -= 1
        res += M
        new = math.ceil(M / 3)
        ext.put(item(new))
        M = ext.queue[0].score
        if len(nums) > 0 and M < nums[len(nums) - 1]:
            M = nums.pop()
        else:
            M = ext.get().score
    return res


def isItPossible(word1: str, word2: str) -> bool:
    w_set1 = set(word1)
    w_set2 = set(word1)
    if (len(w_set2) + len(w_set1)) % 2 != 0:
        return False
    if len(w_set2) == len(word2) and len(w_set1) == len(word1):
        return False
    return True


def prefixCount(words: List[str], pref: str) -> int:
    res = 0
    pref_l = len(pref)
    for w in words:
        if w[0:pref_l] == pref:
            res += 1
    return res


# M1W2
def differenceOfSum(nums: List[int]) -> int:
    str_list = map(str, nums)
    s = "".join(str_list)
    sum1 = 0
    for ch in s:
        sum1 += (ord(ch) - ord('0'))
    return abs(sum(nums) - sum1)


import numpy as np


def rangeAddQueries(n: int, queries: List[List[int]]) -> List[List[int]]:
    res = np.zeros((n * n, 1), int)
    for q in queries:
        for i in range(q[0], q[2] + 1):
            res[n * i + q[1]:n * i + q[3] + 1] += 1
    res = res.reshape((n, n))
    return res.tolist()


class Solution0104:
    def distinctIntegers(self, n: int) -> int:
        return 0 if n == 0 else n - 1

    def monkeyMove(self, n: int) -> int:
        locations = []
        diff = 0
        ans = 0
        for i in range(n):
            l1 = (i + 1) % n
            l2 = (i - 1 + n) % n
            if l1 == l2:
                diff += 1
            locations.append(l1)
            locations.append(l2)
        print(locations)
        la_count = Counter(locations)

        for c in la_count:
            if la_count[c] == 1:
                continue
            ans += 2 ** n - math.comb(la_count[c], 2)* 2**(n-2)
        print(la_count)
        print(diff)
        return ans - diff


if __name__ == '__main__':
    w0104 = Solution0104()
    # res = w0104.distinctIntegers(5)
    res = w0104.monkeyMove(4)
    print(res)
    # res = maximumCount([0])
    # res = maxKelements([0, 100, 0], 5)

    # ext = PriorityQueue()
    # ext.put(item(12))
    # ext.put(item(122))
    # ext.put(item(123))
    # while not ext.empty():
    #     print(ext.get().score)
    # res = differenceOfSum([1, 15, 6, 3])
    # res = rangeAddQueries(2, [[0, 0, 1, 1]])
    # print(res)
