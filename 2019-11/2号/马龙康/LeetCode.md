# LeetCode

 

2.给定一个二叉树，返回它的 后序 遍历。

 

示例:

 

输入: [1,null,2,3]
 1 
 2 / 3

 

输出: [3,2,1]

 

来源：力扣（LeetCode） 链接：https://leetcode-cn.com/problems/binary-tree-postorder-traversa

 

```c
int size(struct TreeNode *root)
{
    if(root == NULL)
        return 0;
    return size(root->left) + size(root->right) +1;
}
void _postorderTraversal(struct TreeNode *root,int *result,int *i)
{
    if(root!=NULL)
    {
        
        _postorderTraversal(root->left,result,i);
        _postorderTraversal(root->right,result,i);
        result[*i] = root->val;
        (*i)++;
    }
}
int* postorderTraversal(struct TreeNode* root, int* returnSize)
{
    int n = size(root);
    int *result = (int*)malloc(sizeof(int)*n);
    *returnSize = n;
    int i = 0;
    _postorderTraversal(root,result,&i);
    return result;
}
```

