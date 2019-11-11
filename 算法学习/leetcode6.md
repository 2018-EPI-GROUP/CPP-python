# 合并两个有序链表
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例：
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```c
struct ListNode* mergeTwoLists(struct ListNode* l1, struct ListNode* l2){
    if(l1 == NULL) return l2;
    if(l2 == NULL) return l1;
    
    struct ListNode newhead, *p1, *p2;
    newhead.next = l1;
    l1 = &newhead;
    
    while(l2 != NULL) {
        p1 = l1->next;
        p2 = l2->next;
        if(p1 == NULL) {
            l1->next = l2;
            break;
        }
        if(l2->val <= p1->val) {
            l2->next = l1->next;
            l1->next = l2;
            l2 = p2;
        }
        l1 = l1->next;
    }
    return newhead.next;
}
```
遍历l2，插入l1对应的位置。
要为l1创建一个虚拟头结点，这样如果l2的最小值小于l1的最小值就可以头插了。

l1链表 ： l1 在左 ， p1 在右 ， l1从虚拟头结点开始， p1紧跟其后。
l2链表 ： l2 在左 ， p2 在右 ， l2从第一个结点开始， p2紧跟其后。

每次比较 l2所指结点的值 与 p1所指结点的值 ， 若满足l2 <= p1 , 则把l2 插到 p1 前面。 然后l2 与 p2 后移一位。
否则 l1 与 p1 后移一位。 

解题参考：
作者：zip-2
来源：力扣（LeetCode）
