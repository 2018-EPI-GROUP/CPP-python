#include "stdio.h"
#define MAX 100
int searchInsert(int* nums, int numsSize, int target) {
	int left = 0;
	int right = numsSize - 1;
	int mid;
	while (left <= right) {
		mid = left + ((right - left) >> 1);//右移操作<<,速度快。此处效果类似于/2.
		if (nums[mid] == target)
		{
			return mid;
		}
		else if (target < nums[mid]) {
			right = mid - 1;
		}
		else if (target > nums[mid]) {
			left = mid + 1;
		}
	}

	if (target < nums[mid]) {
		return mid;
	}
	else {
		return mid + 1;
	}

}

int main()
{
	int i, g, ans, tar;
	int ss[MAX];
	scanf_s("%d", &g);
	for (i = 0; i < g; i++)
	{
		scanf_s("%d", &ss[i]);
	}
	scanf_s("%d", &tar);
	ans = searchInsert(ss, g, tar);
	printf("%d", ans);

}
