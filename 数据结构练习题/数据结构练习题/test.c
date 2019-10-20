1.给定一个带有头结点 head 的非空单链表，返回链表的中间结点。
如果有两个中间结点，则返回第二个中间结点。
示例 1：
输入：[1, 2, 3, 4, 5]
输出：此列表中的结点 3 (序列化形式：[3, 4, 5])
返回的结点值为 3 。(测评系统对该结点序列化表述是[3, 4, 5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
示例 2：
输入：[1, 2, 3, 4, 5, 6]
输出：此列表中的结点 4 (序列化形式：[4, 5, 6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。
来源：力扣（LeetCode）
链接：https ://leetcode-cn.com/problems/middle-of-the-linked-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
//
int GetLength(struct ListNode* head)
{
	int NodeCount = 0;
	struct ListNode* p = head;
	while (p)
	{
		NodeCount++;
		p = p->next;
	}

	return NodeCount;
}

struct ListNode* middleNode(struct ListNode* head)
{
	int NodeCount = GetLength(head);
	int i = 0;
	struct ListNode* p = head;
	for (i = 0; i<NodeCount / 2; ++i)
	{
		p = p->next;
	}
	return p;
}
2.反转一个单链表。

示例 :

输入 : 1->2->3->4->5->NULL
输出 : 5->4->3->2->1->NULL

 进阶 :
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？

来源：力扣（LeetCode）
链接：https ://leetcode-cn.com/problems/reverse-linked-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
struct ListNode* reverseList(struct ListNode* head)
{
	struct ListNode* p;
	struct ListNode* q;
	if (head&&head->next)
	{
		p = head;
		q = p->next;
		p->next = NULL;
		while (q)
		{
			p = q;
			q = q->next;
			p->next = head;
			head = p;
		}
		return head;
	}
	return head;
}
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：


左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。


注意空字符串可被认为是有效字符串。

//栈的例题
bool isValid(char* s)
{

	int length = 0;//定义字符串长度
	while (*(s + length))length++;//获取字符串长度
	char* ptr = (char*)malloc(length / 2);//分配内存空间
	memset(ptr, 0, length / 2);//初始化内存空间
	int i, a = 0;
	for (i = 0; i < length; i++)
	{
		if ((*(s + i) == '(') || (*(s + i) == '{') || (*(s + i) == '['))
		{
			a++;
			*(ptr + a) = *(s + i);
		}
		//'('与')'的ASCII值差1，'['与']'，'{'与'}'的ASCII值差2
		else if ((*(s + i) == (*(ptr + a) + 1)) || (*(s + i) == (*(ptr + a) + 2)))
		{
			a--;
		}
		else return 0;
	}
	if (a)
		return 0;
	return 1;
}
使用栈实现队列的下列操作：


push(x) --将一个元素放入队列的尾部。
pop() --从队列首部移除元素。
peek() --返回队列首部的元素。
empty() --返回队列是否为空。
#define DataType int
#define MAXSIZE  100
typedef struct
{
	DataType *stack_in;
	DataType *stack_out;
	int top_in;
	int top_out;
} MyQueue;
/** Initialize your data structure here. */
MyQueue* myQueueCreate() {
	MyQueue *S = (MyQueue*)malloc(sizeof(MyQueue));
	if (S == NULL)
		return NULL;
	S->stack_in = (int*)malloc(sizeof(DataType)*MAXSIZE);
	S->stack_out = (int*)malloc(sizeof(DataType)*MAXSIZE);
	S->top_in = -1;
	S->top_out = -1;
	return S;
}
/** Push element x to the back of queue. */
void myQueuePush(MyQueue* obj, int x) {
	obj->stack_in[++(obj->top_in)] = x;
}
/** Removes the element from in front of queue and returns that element. */
int myQueuePop(MyQueue* obj) {
	if (obj->top_out == -1)
	{
		while (obj->top_in != -1)
		{
			obj->stack_out[++(obj->top_out)] = obj->stack_in[(obj->top_in)--];
		}
	}
	return obj->stack_out[(obj->top_out)--];
}
/** Get the front element. */
int myQueuePeek(MyQueue* obj) {
	if (obj->top_out != -1)
	{
		return obj->stack_out[(obj->top_out)];
	}
	else if (obj->top_in != -1)
	{
		return obj->stack_in[0];
	}
	else
		return 0;
}
/** Returns whether the queue is empty. */
bool myQueueEmpty(MyQueue* obj) {
	return (obj->top_in == -1 && obj->top_out == -1) ? true : false;
}

void myQueueFree(MyQueue* obj) {
	free(obj);
}
/**
* Your MyQueue struct will be instantiated and called as such:
* MyQueue* obj = myQueueCreate();
* myQueuePush(obj, x);

* int param_2 = myQueuePop(obj);

* int param_3 = myQueuePeek(obj);

* bool param_4 = myQueueEmpty(obj);

* myQueueFree(obj);
*/
//设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。

//循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。//
typedef struct {
	int *que;
	int front;
	int rear;
	int length;
} MyCircularQueue;

/** Initialize your data structure here. Set the size of the queue to be k. */

MyCircularQueue* myCircularQueueCreate(int k) {
	MyCircularQueue *q1;
	q1 = malloc(sizeof(MyCircularQueue));
	q1->que = malloc(sizeof(int)* (k + 1));
	q1->front = q1->rear = 0;
	q1->length = k + 1;
	return q1;
}

/** Insert an element into the circular queue. Return true if the operation is successful. */
bool myCircularQueueEnQueue(MyCircularQueue* obj, int value) {
	if ((obj->rear + 1) % (obj->length) == obj->front)
		return false;
	else
	{
		obj->rear = (obj->rear + 1) % (obj->length);
		obj->que[obj->rear] = value;
		return true;
	}
}

/** Delete an element from the circular queue. Return true if the operation is successful. */
bool myCircularQueueDeQueue(MyCircularQueue* obj) {
	if (obj->rear == obj->front)
		return false;
	else
	{
		obj->front = (obj->front + 1) % (obj->length);
		return true;
	}
}

/** Get the front item from the queue. */
int myCircularQueueFront(MyCircularQueue* obj) {
	if (obj->rear == obj->front)
		return -1;
	return obj->que[(obj->front + 1) % obj->length];

}

/** Get the last item from the queue. */
int myCircularQueueRear(MyCircularQueue* obj) {
	if (obj->rear == obj->front)
		return -1;
	return obj->que[obj->rear];
}

/** Checks whether the circular queue is empty or not. */
bool myCircularQueueIsEmpty(MyCircularQueue* obj) {
	if (obj->rear == obj->front)
		return true;
	return false;
}

/** Checks whether the circular queue is full or not. */
bool myCircularQueueIsFull(MyCircularQueue* obj) {
	if ((obj->rear + 1) % (obj->length) == obj->front)
		return true;
	return false;
}

void myCircularQueueFree(MyCircularQueue* obj) {
	int *tmp;
	tmp = obj->que;
	obj->que = NULL;
	free(tmp);
}
