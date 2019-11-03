# LeetCode

 

3.给定一个二叉树，返回它的中序 遍历。

 

示例:

 

输入: [1,null,2,3] 1 
 2 / 3

 

输出: [1,3,2]

 

来源：力扣（LeetCode） 链接：https://leetcode-cn.com/problems/binary-tree-inorder-traversal

 

```c
int size(struct TreeNode *root)
{
    if(root == NULL)
        return 0;
    return size(root->left)+size(root->right)+1;
}
void _inorderTraversal(struct TreeNode *root,int *result,int *i)
{
    if(root!=NULL)
    {
    _inorderTraversal(root->left,result,i);
    result[*i] = root->val;
    (*i)++; 
    _inorderTraversal(root->right,result,i);
    }
}
int* inorderTraversal(struct TreeNode* root, int* returnSize)
{
    int n = size(root);
    int *result = (int*)malloc(sizeof(int)*n);
    *returnSize = n;
    int i = 0;
    _inorderTraversal(root,result,&i);
    
    return result;

}
```