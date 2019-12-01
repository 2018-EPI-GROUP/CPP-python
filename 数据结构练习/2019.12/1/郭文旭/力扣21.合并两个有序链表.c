/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */


struct ListNode* mergeTwoLists(struct ListNode* l1, struct ListNode* l2){
    int temp;
    if (l1 == NULL)
		return l2;
	if (l2 == NULL)
		return l1;
	struct ListNode* p, * q,*d;
    if(l1->val<=l2->val)
    {	
    p = l1;
    d=p;
    p=p->next;
	q = l2;
       
	while(d->next!=NULL&&l2!=NULL)
		if (p->val >= l2->val)
		{
			l2 = l2->next;
			q->next = d->next;
			d->next = q;
            d=q;
			p = d->next;
			q = l2;
		}
		else
		{
            d=p;
			p = p->next;
		}
	if (d->next == NULL)
		d->next = l2;

        return l1;
    }
    else
    {
    p = l2;
    d=p;
    p=p->next;
	q = l1;
    while(d->next!=NULL&&l1!=NULL)
		if (p->val >= l1->val)
		{
			l1 = l1->next;
			q->next = d->next;
			d->next = q;
            d=q;
			p = d->next;
			q = l1;
		}
		else
		{
            d=p;
			p = p->next;
		}
	if (d->next == NULL)
		d->next = l1;

        return l2;
    }

    
}
