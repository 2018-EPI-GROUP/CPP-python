# 最后一个单词的长度
给定一个仅包含大小写字母和空格 ' ' 的字符串，返回其最后一个单词的长度。

如果不存在最后一个单词，请返回 0 。

说明：一个单词是指由字母组成，但不包含任何空格的字符串。

```c
int lengthOfLastWord(char * s){
    int lenth=0;
    int count=0;
    //求字符串长度
    while (*(s+lenth)!='\0') {
        lenth++;
    }
    if (lenth==0) {
        return 0;
    }
    
    int i;
    int flag=0;
    for (i=lenth-1;i>=0;i--) {
        if (*(s+i)==' ' && flag==0) {
            
        } else if (('a'<=*(s+i)  &&  *(s+i)<='z') || ('A'<=*(s+i) && *(s+i)<='Z')) {
            count++;
            flag=1;
        } else if (*(s+i)==' ' && flag==1) {
            return count;
        }
    }
    return count;
}
```
解题参考：
作者：sunpf
来源：力扣（LeetCode）