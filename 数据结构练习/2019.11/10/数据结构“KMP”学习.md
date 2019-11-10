## C语言学习日志

<!--根据数据结构课本学习KMP算法-->

### 一.BF算法 

#### 1.算法思想

​	【BF算法思想】：蛮力匹配算法，从主串中第pos个字符开始，和模式串T的第一个字符开始比较，若相等，继续比较后续字符；否则回到主串中第pos+1个字符开始重新和模式串T开始比较。直到模式串中有与主串所有字符都匹配，则称模式匹配成功，否者不成功。

#### 2.代码实现

```c
int Index(SSteing S,int pos,SString T)
{
    int i=pos,j=1;					//主串从pos位置开始，模式串从第一个位置开始
    while(i<=S.len&&j<=T.len)
    {
        if(S.ch[i]==T.ch[j])		//匹配字符
        {							//成功匹配下一个字符
            i++;
            j++;
        }
        else						//失败从主串下一个字符重新匹配
        {
            i=i-j+2;
            j=1;
        }
    }
    if(j>T.len)
        return i-T.len;
    else
        return 0;
}
```

### 二.KMP算法

#### 1.算法思想

​	在KMP算法中，主串与模式串比较，每当出现字符比较不等式，主串中的 i 指针不需要回溯，根据已经得到的部分匹配结果将模式串尽可能的向右滑动，然后继续进行比较。

#### 2.代码实现

##### （1）主体算法部分

```c
int (SString S,int pos,SString T，int next[ ])
{
    int i=pos,k=1;
    Get_Next(T,next);
    while(i<=S.len&&k<=T.len)
    {
        if(k==0||S.ch[i]==T.ch[k])		//逐级向后比较
        {
            i++;
            k++;
        }
        else
        {
            j=next[k];					//通过next数组来确定比较的位置
        }
        if(j<T.len) 
            return i-T.len;				//匹配成功，返回匹配的初始位置
        else 
            return 0;
    }
}
```

##### （2）next算法部分

KMP算法是已知next[]数组的值，下面为求next[]数组值的代码实现

```c
void Get_Next(SString T,int next[ ])
{
    int j=1,k=0;
    next[1]=0;
    while(j<T.len)
    {
        if(k==0||T.ch[j]==T.ch[k])		//若发现前缀，后缀相同的部分，k递增
        {
            k++;
            j++;
            next[j]=k;
        }
        else
            k=next[k];			//若前后缀不同，返回刚才索引的前缀的前数值，查看是否相同
    }
}
```

##### （3）nextval算法部分

​	虽然next能帮助减少许多不必要的判断过程，但还是会有些情况下尚有缺陷。提出nextval函数帮助next函数减少不必要的重复比较。

```c
int Get_Nextval(SString T,int next[ ],int nextval[ ])
{
    int j=2,k=0;
    Get_Next(T,next)
    nextval[1]=0;
    while(j<=T.len)
    {
        k=next[j];
        if(T.ch[j]==T.ch[k])
            nextval[j]=nextval[k];
        else 
            nextval[j]=next[j];
        j++;         
    }
}
```

