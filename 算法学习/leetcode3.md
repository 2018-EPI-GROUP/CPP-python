# 回文数
判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
- python实现
```python
    def isPalindrome(self, x: int) -> bool:
        true = True
        false = False   
        x = int(x)
        if x>=0:
            pass
        else:
            return false
        y = str(x)
        n = len(y)
        sum = []
        num = y[::-1]
        if (num == y):
            return true
        else:
            return false
```
执行用时 :68 ms
内存消耗 :13.9 MB
- python改进
```python
#方法一：将变量整体做字符串处理，因为不相同既非回文数，不用判断正负
class Solution:
    def isPalindrome(self, x: int) -> bool:        
        if str(x) == str(x)[::-1]:
            return True
        else:
            return False

#方法二：使用整数处理
class Solution:
    def isPalindrome(self, x: int) -> bool:        
        if x < 0 :
            return False
        
        m,n = x,0
        
        while m:
            n = n*10 + m%10
            m = m//10
            
        if x == n:
            return True
        else:
            return False
```
答案参考：

作者：tu-dou-G6CIO5IOm8
来源：力扣（LeetCode）

- c语言实现
```c
bool isPalindrome(int x){
    int temp;
    long y=0;
    int start=x;//判断正负
    if(x<0)
    {
        return false;
    }
    else
    {
        while(x!=0)
        {
            temp=x%10;//每次取余数
            x=x/10;
            y=temp+y*10;
        }
        if(start == y)
        {
            return true;
        }
        return false;
    }
}
```
答案参考：

作者：mia-4
来源：力扣（LeetCode）