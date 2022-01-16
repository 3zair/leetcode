"""
# 20220116
# 水塘抽样问题
# 核心是每次取随机数都遍历一遍，使用时间换空间
# 适用于在一个大容量未知的待选池中随机选出m个样本

## 复杂度：
### 空间O(1)
### 时间O(n)
"""

import random


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    head = ListNode()

    def __init__(self, head):
        self.head = head

    def getRandom(self):
        node = self.head
        i = 1
        ans = node.val
        while node is not None:
            if random.randrange(i) == 0:
                ans = node.val
            node = node.next
            i += 1
        return ans
