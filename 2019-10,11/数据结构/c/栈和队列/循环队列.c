

/*队列结构体定义*/
typedef struct {
    int *val;
    int front,rear,max;
} MyCircularQueue;
bool myCircularQueueIsEmpty(MyCircularQueue* obj);
bool myCircularQueueIsFull(MyCircularQueue* obj);
/*创建长度为k的循环队列*/

MyCircularQueue* myCircularQueueCreate(int k) {
    MyCircularQueue *x;
    x = (MyCircularQueue *)malloc(sizeof(MyCircularQueue));
    x->val = (int*)malloc(sizeof(int)*k);
    x->front = -1;
    x->rear = -1;
    x->max  = k;
    return x;
}

/*向队列obj中插入 value */
bool myCircularQueueEnQueue(MyCircularQueue* obj, int value) {
    if(myCircularQueueIsFull(obj)){
        return 0;
    }
    else{
        if(myCircularQueueIsEmpty(obj)){
            obj->front = 0;
            obj->rear = 0;
            *(obj->val+obj->rear) = value;
        }
        else{
            obj->rear = (obj->rear+1)%(obj->max);
            *(obj->val+obj->rear) = value;
        }
    }
    return 1;
}

/* 删除队列中头元素 */
bool myCircularQueueDeQueue(MyCircularQueue* obj) {
    if(myCircularQueueIsEmpty(obj)){
        return 0;
    }
    if(obj->front == obj->rear){
        obj->rear = -1;
        obj->front = -1;
    }    
    else{
        obj->front = (obj->max + obj->front+1)%(obj->max);
    }
    return 1;
}

/* 获得队列中的头元素 */
int myCircularQueueFront(MyCircularQueue* obj) {
    if(myCircularQueueIsEmpty(obj)) return -1;
    return *(obj->val+obj->front);
}

/* 获得队列中的尾元素 */
int myCircularQueueRear(MyCircularQueue* obj) {
    if(myCircularQueueIsEmpty(obj)) return -1;
    return *(obj->val+obj->rear);
}

/* 判断队列是否为空 */
bool myCircularQueueIsEmpty(MyCircularQueue* obj) {
    if(obj->front == -1){
        return true;
    }
    return false;
}

/* 判断队列是否为满 */
bool myCircularQueueIsFull(MyCircularQueue* obj) {
    if(obj->front == (obj->rear+1)%(obj->max)){
        return true;
    }
    return false;
}
/*释放队列*/
void myCircularQueueFree(MyCircularQueue* obj) {
    free(obj->val);
    free(obj);
}
