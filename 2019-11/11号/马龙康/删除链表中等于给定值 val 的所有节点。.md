删除链表中等于给定值 ***val\*** 的所有节点。

 

**示例:**

 1

重复元素1

```
输入: 1->2->6->3->4->5->6, val = 6
输出: 1->2->3->4->5
```

```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */


struct ListNode* removeElements(struct ListNode* head, int val)
{
    if(head == NULL)
    return NULL;
    struct ListNode *p=head,*temp = NULL;
    while(head->val == val)
    {
        if(head->next == NULL)
        {
            free(head);
            return NULL;
        }
        temp = head->next;
        head->val = head->next->val;
        head->next = head->next->next;
        free(temp);
    }//判断表头只有一个元素，或者两个元素时候,也算重复必须删除
    while(p->next)
    {
        if(p->next->val == val)
        {
            temp = p->next;
            p->next = p->next->next;
            free(temp);
        }
        else
        p = p->next;
    }
    return head;
}
```

