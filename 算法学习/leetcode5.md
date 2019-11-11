# 删除排序链表中的重复元素
给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

示例 1:
输入: 1->1->2
输出: 1->2

示例 2:
输入: 1->1->2->3->3
输出: 1->2->3
```c
struct ListNode* deleteDuplicates(struct ListNode* head){
    struct ListNode *p,*q;
    if(!head) return NULL;    
    p=head;
    q=head->next;
    while(q)
    {
      if(head->val==q->val)
        {
            head->next=q->next;
            free(q);
            q=head->next;
        }
      else
      {
          head=head->next;
          q=q->next;
      }
    }
    head=p;
    return head;
   }
```
解题参考：
作者：cocowy
来源：力扣（LeetCode）