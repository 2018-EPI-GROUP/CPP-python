## MP3光标位置

```C++

#include <iostream>

using namespace std;

int main()

{

    int n;//歌曲个数

    while(cin>>n)

    {

        string s;//操纵字符串

        cin>>s;

 

        int top=0;//保存当前列表顶部歌曲

        int button=0;//保存当前列表底部歌曲

        int cur=1;//保存当前光标指向歌曲数

 

        if(n<=4)

        {

            //初始化三个变量

            top=1;

            button=n;

            //cur=1;

 

            int i=0;

            while(i<s.size())

            {

                if(s[i]=='U')//上键

                {

                    if(cur==top)//如果当前光标在第首部

                    {

                        cur=button;

                    }

                    else

                    {

                        cur--;

                    }

                }

                else if(s[i]=='D')//下键

                {

                    if(cur==button)//当前光标在底部

                    {

                        cur=1;

                    }

                    else

                    {

                        cur++;

                    }

                }

                i++;

            } 

        }

        else//歌曲数大于4

        {

            top=1;

            cur=1;

            button=4;

 

            int i=0;

            while(i<s.size())

            {

                if(s[i]=='U')//上键

                {

                    if(cur==1)//当前光标在1

                    {

                        cur=n;

                        button=n;

                        top=button-3;

                    }

                    else if(cur==top)//当前光标在当前顶部，需要翻页

                    {

                        cur--;

                        top--;

                        button--;

                    }

                    else//只需要移动光标

                    {

                        cur--;

                    }

                }

                else if(s[i]=='D')//下键

                {

                    if(cur==n)//光标在最后一个

                    {

                        cur=1;

                        top=1;

                        button=4;

                    }

                    else if(cur==button)//光标在当前底部，需要翻页

                    {

                        cur++;

                        top++;

                        button++;

                    }

                    else//只需要移动光标

                    {

                        cur++;

                    }

                }

                i++;

            }

        }

 

        //输出结果

        for(int i=top;i<button;i++)

        {

            cout<<i<<" ";

        }

        cout<<button<<endl;

        cout<<cur<<endl;

    }

    return 0;

}

```

