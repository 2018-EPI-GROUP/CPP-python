# C语言学习日志

<!--根据 **leetcode**做题问题及解决方案-->

### “两数之和”

> 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
>
> 你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

#### 代码块

方法一：暴力法

```c
/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* twoSum(int* nums, int numsSize, int target, int* returnSize)
{
    int i, j, shu1, shu2;

    *returnSize = 0;
    int* arr = (int*)malloc(sizeof(int) * 2);
	for (i = 0; i < numsSize; i++)
	{
		shu1 =  nums[i];
		for (j = i + 1; j < numsSize; j++)
		{
			shu2 = nums[j];
			if (shu1 + shu2 == target)
			{
				printf("%d %d", i, j);				
				arr[0] = i;
				arr[1] = j;
				*returnSize = 2;
                return arr;
			}
			
		}
	
		
	}
    return arr;
}
/*
时间复杂度：O（n平方）
空间复杂度：O（1）
耗时：244ms
*/
```

自己完成了简单的暴力法

但代码并不够完美

方法二：哈希表

> 为了对运行时间复杂度进行优化，我们需要一种更有效的方法来检查数组中是否存在目标元素。如果存在，我们需要找出它的索引。保持数组中的每个元素与其索引相互对应的最好方法是什么？哈希表。
>
> 通过以空间换取速度的方式，我们可以将查找时间从 O(n) 降低到 O(1)。哈希表正是为此目的而构建的，它支持以 近似 恒定的时间进行快速查找。我用“近似”来描述，是因为一旦出现冲突，查找用时可能会退化到 O(n)。但只要你仔细地挑选哈希函数，在哈希表中进行查找的用时应当被摊销为 O(1)。
>
> 一个简单的实现使用了两次迭代。在第一次迭代中，我们将每个元素的值和它的索引添加到表中。然后，在第二次迭代中，我们将检查每个元素所对应的目标元素（target−nums[i]）是否存在于表中。注意，该目标元素不能是 nums[i] 本身！
>
> ```c
> struct hash_data{
>     int key;
>     int data;
>     struct hash_data * next;
> };
> 
>  struct hash_table
> {
>     struct hash_data ** head; //数组
>     int hash_width;
> };
> 
> ///初始化
> int hash_init(struct hash_table * table, int width){
>     if(width<=0)
>         return -1;
>     struct hash_data **tmp = malloc(sizeof(struct hash_data *)*width);
>     table->head = tmp;
>     memset(table->head, 0, width * sizeof(struct hash_data *));
>     if(table->head==NULL)
>         return -1;
>     table->hash_width = width;
>     return 0;
> }
> 
> ///释放
> void hash_free(struct hash_table table){
>     if(table.head!=NULL){
>         for (int i=0; i<table.hash_width; i++) {
>             struct hash_data* element_head= table.head[I];
>             while (element_head !=NULL) {
>                 struct hash_data* temp =element_head;
>                 element_head = element_head->next;
>                 free(temp);
>             }
>         }
>         free(table.head);
>         table.head = NULL;
>     }
>     table.hash_width = 0;
> }
> 
> int hash_addr(struct hash_table table,int key){
>     int addr =abs(key) % table.hash_width;
>     return addr;
> }
> 
> ///增加
> int hash_insert(struct hash_table table,int key, int value){
>     struct hash_data * tmp = malloc(sizeof(struct hash_data));
>     if(tmp == NULL)
>         return -1;
>     tmp->key = key;
>     tmp->data = value;
>     int k = hash_addr(table,key);
>     tmp->next =table.head[k];
>     table.head[k]=tmp;
>     return 0;
> }
> 
> ///查找
> struct hash_data* hash_find(struct hash_table table, int key){
>     int k = hash_addr(table,key);
>     struct hash_data* element_head=table.head[k];
>     while (element_head !=NULL) {
>         if ( element_head->key == key) {
>             return element_head;
>         }
>         element_head = element_head->next;
>     }
>     return NULL;
> }
> 
> int* twoSum(int* nums, int numsSize, int target, int* returnSize){
>       int* res = (int *)malloc(sizeof(int) * 2);
>     * returnSize=0;
>     struct hash_table table;
>     hash_init(&table, 100);
>     for(int i = 0; i < numsSize; I++)
>     {
>       int value = target - nums[I];
>     struct hash_data* data=  hash_find(table, value);
>         if (data !=NULL && data->data != i) {
>             res[1]=I;
>             res[0]=data->data;
>             * returnSize=2;
>             break;
>         }
>         hash_insert(table,nums[i] ,i);
>     }
>     hash_free(table);
>     return res;
> }
> 
> 作者：chen-xing-15
> 链接：https://leetcode-cn.com/problems/two-sum/solution/liang-shu-zhi-he-san-chong-jie-fa-by-chen-xing-15/
> 来源：力扣（LeetCode）
> 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
> ```

从解题中看见大佬写的哈希表，进行学习，但并没有完全理解，还在努力中！！！

