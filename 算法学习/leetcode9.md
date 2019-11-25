# 相同的树
给定两个二叉树，编写一个函数来检验它们是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

示例 1:
```
输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true
```
示例 2:
```
输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false
```
- 递归法求解
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
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(p==NULL&&q==NULL) return true;
        else if (p==NULL||q==NULL) return false;
//比较子节点的父节点，避免在此处分叉，如果相同不能判定两树相同，不同可判定两数不同
        if(p->val != q->val) return false;
//递归求解左子树和右子树
        if(!isSameTree(p->left,q->left)) return false;
        if(!isSameTree(p->right,q->right)) return false;
//都满足则判定相同
        return true;
    }
};
```