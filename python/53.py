# 53

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max = nums[0]
        for i in range(1, len(nums)):
            # 更新左端点， 如果当前已经是负的了，则重新立i为起点
            if nums[i] + nums[i - 1] > nums[i]:
                nums[i] += nums[i - 1]
            # 更新又断点
            if max < nums[i]:
                max = nums[i]

        return max
