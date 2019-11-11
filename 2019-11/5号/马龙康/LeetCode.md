# LeetCode

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7

返回它的最大深度 3 。

```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */


int maxDepth(struct TreeNode* root)
{
    if(root == NULL)
        return 0;
    else
    {
        int left_h = maxDepth(root->left);
        int right_h = maxDepth(root->right);
        return (left_h>right_h?left_h:right_h)+1;//加上根节点
    }

}
```

