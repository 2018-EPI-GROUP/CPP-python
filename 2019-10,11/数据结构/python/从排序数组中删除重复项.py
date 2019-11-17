class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        a = nums[0]-1
        b = []
        for i in nums:
            if a == i:
                continue
            else:
                b += [i]
                a = i
        for i in range(len(b)):
            nums[i] = b[i]
        return len(b)