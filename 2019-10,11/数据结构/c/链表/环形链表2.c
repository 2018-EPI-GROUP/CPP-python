/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */
struct ListNode *detectCycle(struct ListNode *head) {
    if(head == NULL||head->next == NULL){
        return NULL;
    }
    struct ListNode *q,*p;
    p = head;
    q = head;
    int i=0,j=0,flag=0,index=1;
    while(q){
        index++;
        q = q->next;
        if(q == p||q ==NULL){
            break;
        }
        q = q->next;
        if(q == p||q == NULL){
            break;
        }
        p=p->next;
    }
    if(q==NULL) return NULL;
    q = head;
    p = head;
    while(1){
        for(i = 0;i < index;i++){
            q = q->next;
            if(q == p){
                flag = 1;
                break;
            }
        }
         if(flag){
            break;
        }
        p = p->next;
        j++;
    }
    return p;
}

//该代码在力扣中运行200ms
//以下为0ms代码：
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */
struct ListNode *detectCycle(struct ListNode *head) {
    struct ListNode *slow, *fast;
    bool hasCycle = false;
    slow = fast = head;
    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;

        if (slow == fast) {
            hasCycle = true;
            break;
        }
    }

    if (hasCycle) {
        slow = head;

        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }

    return NULL;
}