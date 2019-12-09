# 杨辉三角
给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
在杨辉三角中，每个数是它左上方和右上方的数的和。

示例:
```
输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```
- 解题：
```C++
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> ans(numRows);//numRows赋值给ans
        if(numRows == 0)    return ans;//空返回空
        for(int i = 0; i < numRows; ++ i )//向量实现二维数组
        {
            for(int j = 0; j <= i; ++ j)//杨辉三角的列数等于行数
            {
                if(j == 0 || j == i) 
                    ans[i].push_back(1);//首尾都是1
                else
                    ans[i].push_back(ans[i-1][j-1] + ans[i-1][j]); //其他位等于左上和右上的和
            }
        }
        return ans;//返回向量
    }
};
```