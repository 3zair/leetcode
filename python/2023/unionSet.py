from typing import List


class UnionSet128:
    def __init__(self, nums):
        self.leftMap = {}
        self.rightMap = {}
        self.maxSize = 1
        for n in nums:
            self.leftMap[n] = n
            self.rightMap[n] = n

    def find(self, a):
        return a in self.leftMap

    def findLeftHead(self, a):
        tmpList = []
        while self.leftMap[a] != a:
            tmpList.append(a)
            a = self.leftMap[a]
        for aa in tmpList:
            self.leftMap[aa] = a
        return a

    def findRightHead(self, a):
        tmpList = []
        while self.rightMap[a] != a:
            a = self.rightMap[a]
        for aa in tmpList:
            self.rightMap[aa] = a
        return a

    # def isSameSet(self, a, b):
    #     return self.findLeftHead(a) == b and self.findRightHead(a) == b

    def union(self, a, b):
        aLeft = self.findLeftHead(a)
        aRight = self.findRightHead(a)
        bLeft = self.findLeftHead(b)
        bRight = self.findRightHead(b)
        if aLeft == bLeft and aRight == bRight:
            return
        if (aLeft <= bLeft <= aRight + 1) or (bLeft <= aLeft <= bRight + 1):
            newLeft = min(aLeft, bLeft)
            newRight = max(aRight, bRight)
            self.leftMap[aLeft] = newLeft
            self.leftMap[bLeft] = newLeft
            self.rightMap[aRight] = newRight
            self.rightMap[bRight] = newRight
            self.maxSize = max(newRight - newLeft + 1, self.maxSize)

    def getMaxSize(self, ):
        return self.maxSize


class UnionSetSolution:
    # 128 https://leetcode.cn/problems/longest-consecutive-sequence/
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        u128 = UnionSet128(nums)
        for i in range(len(nums)):
            if u128.find(nums[i] + 1):
                u128.union(nums[i], nums[i] + 1)
        return u128.getMaxSize()


if __name__ == '__main__':
    S = UnionSetSolution()
    print(S.longestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))
