# 最长公共前缀

编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 `""`。
```c
//定义函数
char * longestCommonPrefix(char ** strs, int strsSize){
//如果读入空值，函数返回空字符串
	if (strsSize == 0) {
        char *ret = (char *)malloc(1);
        ret[0] = '\0';
        return ret;
    }
//如果字符长度都是一，返回首字母
    if (strsSize == 1) return strs[0];
//遍历查找，查找到最小值的长度之前
	int i = 0, j, is = 1, tmp;
    for (; is; ++i) {
        tmp = strs[0][i];
        for (j = 0; j  < strsSize; ++j) 
            if ((!strs[j][i]) || (strs[j][i] != tmp)) 
                {   is = 0;     break;  }
    }
    strs[0][i-1] = '\0';
    return strs[0];
}
```

答案参考：

作者：ljj666
来源：力扣（LeetCode）