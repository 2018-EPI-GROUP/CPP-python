# 删除排序链表中的重复元素
给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
示例 1:
```
输入: 1->1->2
输出: 1->2
示例 2:

输入: 1->1->2->3->3
输出: 1->2->3
```
- 解题：
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
 class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(!head||!head->next)//链表空或链表指向空返回
            return head;
        ListNode* p=head;
        while(p->next!=NULL&&p!=NULL)
        {
            if(p->val == p->next->val)
            {
                p->next=p->next->next;
            }
            else 
                p=p->next;
        }
        return head;
    }
};
```