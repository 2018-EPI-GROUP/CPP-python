/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */


struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2){
    int x = 0;
    if(l1->val == 0&&l1->next == NULL){
        return l2;
    }
    if(l2->val == 0&&l2->next == NULL){
        return l1;
    }
    struct ListNode *q1,*q2,*l3,*q3;
    l3 = (struct ListNode *)malloc(sizeof(struct ListNode));
    q1 = l1;
    q2 = l2;
    q3 = l3;
    while(q1 != NULL && q2 != NULL){
        q3->next = (struct ListNode *)malloc(sizeof(struct ListNode));
        
       
        if(q1->val + q2->val+x > 9){
            q3->next->val = (q1->val+q2->val+x)%10;
            x = 1;
        }
        else{
            q3->next->val = q1->val+q2->val+x;
            x = 0;
        }
        q3 = q3->next;
        q3->next = NULL;
        q2 = q2->next;
        q1 = q1->next;
    }
    if(x==1&&q1 == NULL&& q2 == NULL){
        q3->next = (struct ListNode *)malloc(sizeof(struct ListNode));
        q3->next->val = 1;
        q3 = q3->next;
        q3->next = NULL;
        x = 0;
    }
    if(q1 != NULL){
         q3->next = q1;
        while(x){
                if(q3->next == NULL && x ==1){
                    
                    q3->next = (struct ListNode *)malloc(sizeof(struct ListNode));
                    q3->next->val = 1;
                    q3 = q3->next;
                    q3->next = NULL;
                    x = 0;
                }
                else{
                    if(q3->next->val + x>9){
                        q3->next->val = (q3->next->val+x)%10;
                        x = 1;
                    }
                    else{
                        q3->next->val = q3->next->val+x;
                        x = 0;
                    }
                    q3 = q3->next;
                }
                
            };
       
    }
    if(q2 != NULL){
        q3->next = q2;
        if(x){
            while(x){
                if(q3->next == NULL && x ==1){
                    
                    q3->next = (struct ListNode *)malloc(sizeof(struct ListNode));
                    q3->next->val = 1;
                    q3 = q3->next;
                    q3->next = NULL;
                    x = 0;
                }
                else{
                    if(q3->next->val + x>9){
                        q3->next->val = (q3->next->val+x)%10;
                        x = 1;
                    }
                    else{
                        q3->next->val = q3->next->val+x;
                        x = 0;
                    }
                    q3 = q3->next;
                }
                
            };
        }
    }
    return l3->next;
    
    
    
    
}