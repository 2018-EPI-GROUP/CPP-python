# 二叉树的最小深度
给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明: 叶子节点是指没有子节点的节点。

示例:

给定二叉树 [3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
```
返回它的最小深度  2.
- 解题：
```python
class Solution:
    def minDepth(self, root):

        if not root: 
            return 0 
#最小深度不同于最大深度，只要有叶即停止递归        
        children = [root.left, root.right]
        if not any(children):
            return 1
#要找到最小值要比对递归项和现有项
        min_depth = float('inf')
        for c in children:
            if c:
                min_depth = min(self.minDepth(c), min_depth)
        return min_depth + 1 

```