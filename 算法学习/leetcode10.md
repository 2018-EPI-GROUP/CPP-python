# 对称二叉树
给定一个二叉树，检查它是否是镜像对称的。
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
```
    1
   / \
  2   2
   \   \
   3    3
```
解决方案：
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        return ismirror(root,root);
    }
    bool ismirror(TreeNode* root_l,TreeNode* root_r)
    {
        if(root_l == NULL && root_r == NULL)//都为空
            return true;
        if(root_l == NULL || root_r == NULL)//有一个为空
            return false;
        return (root_l->val==root_r->val)&&ismirror(root_l->left,root_r->right)&&ismirror(root_l->right,root_r->left);
    }
};
```
问题：在第二个函数中必须把root_l和root_r指向NULL的所有情况列出，否则会出现运行时错误：
runtime error: member access within null pointer of type 'struct TreeNode' (solution.cpp)