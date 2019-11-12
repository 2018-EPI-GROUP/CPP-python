# Cpp-python
C++/python组任务提交仓库
一.字符串

   1.串类型定义

  串(String)是由零个或多个字符组成的有限序列。一般记作S=‘a1a2a3…an’，其中S 是串名，单引号括起来的字符序列是串值；

  ai(1≦i≦n)可以是字母、数字或其它字符；串中所包含的字符个数称为该串的长度。

  空串(Empty String)：长度为零的串。它不包含任何字符。

  子串：串中任意个连续字符组成的子序列。 

  主串：包含子串的串。通常将子串在主串中首次出现时的该子串的首字符对应的主串中的序号，定义为子串在主串中的序号（或位置）。

  串的基本操作集：StrLength,StrAssign,SteConcat,SubStr,StrCmp,StrIndex,StrInsert,StrDelete,StrRep.

   2.串的表示和实现

   （1）串的定长顺序存储结构

      #define MAXSIZE 256

      char s[MAXSIZE];

      typedef struct

      { char data[MAXSIZE];

        int curlen;

      } SeqString;

      SeqString s;

    3.串的链式存储结构（带头结点的单链表）

    结点的类型定义：

    Typedef struct node

    { char data;     //存放字符

      Struct node * next;      //指针域

      }Linkstring；

    4.链式串基本运算算法

    （1）赋值运算

        将一个C/C++字符数组t赋给串s.

     Void StrAssign (LinkString * &s, char t[])

     { int i= 0;

       LinkString *q, * tc;

       s= (Linkstring * ) malloc (sizeof (LinkString));    //建立头结点

       s- > next = NULL;     //初始化链表指针

       tc= s;

       while (t[i]!= '\0')    //将整个串逐个字符地申请结点并建立链表及相应的值

      { q= (LinkString *)malloc (sizeof(LinkString));

        q-> data= t[i];

        tc- > next= q; tc= q;

        i++ ;

      }

      tc-> next = NULL; 

      }

      (2)串连接运算

      将串t连接到串s之后,返回其结果。

      LinkString * Concat( LinkString* s, LihkString * t)

      { LinkString * p= s-> next, * q, * tc,* r;

        r= (LinkString * )malloc (sizeof(LinkString));   //建立头结点

        r- > next = NULL;

        tc= r;    //tc总是指向新链表的最后-个结点

        while (p!= NULL)   //将s串复制给r

      { q= (LinkString * ) malloc (sizeof (LinkString));

        q-> data= P-> data;

        tc->next=q;tc=q;

        P= P- > next;

        }

        p= t- > next;

        while (p!= NULL)   //将t串复制给r

      { q= (LinkString兴)malloc (sizeof (LinkString) ) ;

        q-> data= p-> data;

        tc-> next= q; tc= q;

        P= P- > next;

       }tc- > next = NULL;

        return(r) ;

      }

      

      

二.树

1.树的定义

     树是由n（n>=0）个结点的有穷集合，其中：（1）每个元素称为结点(node)；（2）有一个特定的结点被称为根结点或树根（root);

(3)除根节点以外的其余数据元素被分为m个互不相交的集合T1，T2...Tm-1,其中每一个集合Ti本身也是一颗树，被称作原树的字树（subtree).

2.树的基本术语

结点---表示树中的元素，包括数据项及若干指向其子树的分支.

结点的度---结点拥有的子树数.

叶子---度为0的结点， 也叫终端结点.

分支结点---度不为0的结点，也叫非终端结点.

内部结点--- 除根结点之外， 分支结点也称为内部结点.

树的度---一棵树中最大的结点度数.

孩子---结点子树的根称为该结点的孩子.

双亲---孩子结点的上层结点叫该结点的双亲.

兄弟---同一双亲的孩子之间互成为兄弟.

祖先--- 结点的祖先是从根到该结点所经分支上的所有结点.

子孙--- 以某结点为根的子树中的任一结点都成为该结点的子孙.

结点的层次---从根结点算起，根为第一层， 它的孩子为第二层...

深度---树中结点的最大层次数.

有序树---如果将树中结点的各子树看成从左至右是有次序的(即不能互换)，则称该树为有序树,否则称为无序村。在有序树中最左边的子树的根称

为第一一个孩子，最右边的称为最后一个孩子.

森林---m(m>=0)棵互不相交的树的集合.

3.树的基本操作

(1)双亲结点parent(2)左孩子结点leftChild(3)右兄弟结点rightSibling(4)遍历树traverse(vs)

4.树的存储结构

（1）树的存储表示方法

双亲表示法，孩子表示法，双亲孩子表示法，孩子兄弟表示法等。

孩子兄弟表示法存储结构：

typedef struct TreeNode{

      elemtype data;

      struct TreeNode * son;

      struct TreeNode * next;

      }NodeType;

5.树的遍历

（1）先根遍历

 树的先根遍历递归算法为:

1)访问根结点;2)按照从左到右的次序先根遍历根结点的每一棵子树。

（2）后根遍历

 树的后根遍历递归算法为: 

 1)按照从左到右的次序后根遍历根结点的每一棵子树;2) 访问根结点。

（3）层次遍历

  从树的第一层（根节点）开始，从上至下逐层遍历，在同一层中，则按从左到右的顺序对结点逐个访问。
  
  
  一：二叉树
1.定义：一棵二叉树是结点的一个有限集合，该集合或者为空，或者是由一个根节点加上两棵别称为左子树和右子树的二叉树组成。

2.二叉树的特点：
   a.每个结点最多有两棵子树，即二叉树不存在度大于2的结点。
   b.二叉树的子树有左右之分，其子树的次序不能颠倒。

3.二叉树的基本性质
(1). 在二叉树的第ｉ（ｉ>=１）层最多有２＾(ｉ - １)个结点。
(2). 深度为k(k>=0)的二叉树最少有k个结点，最多有２＾ｋ－１个结点。
(3). 对于任一棵非空二叉树，若其叶结点数为n0，度为2的非叶结点数为n2，则ｎ0 = ｎ2 ＋１。
(4). 具有n个结点的完全二叉树的深度为int_UP（log(2，ｎ+1)）。
(5). 如果将一棵有n个结点的完全二叉树自顶向下，同一层自左向右连续给结点编号１，２，３，．．．．．．，ｎ，然后按此结点编号将树中各结点顺序的存放于一个一维数组，并简称编号为i的结点为结点i（ ｉ>=１ && ｉ<=ｎ）,则有以下关系：
a.若 ｉ= 1，则结点i为根，无父结点；若 ｉ> 1，则结点 i 的父结点为结点int_DOWN（ｉ / ２）;
b.若 ２＊ｉ <= ｎ，则结点 ｉ 的左子女为结点 ２＊ｉ；
c.若２＊ｉ＜＝ｎ，则结点ｉ的右子女为结点２＊ｉ＋１；
d.若结点编号ｉ为奇数，且ｉ！＝１，它处于右兄弟位置，则它的左兄弟为结点ｉ－１；
e.若结点编号ｉ为偶数，且ｉ！＝ｎ，它处于左兄弟位置，则它的右兄弟为结点ｉ＋１；

4.二叉树的存储方式
二叉树一般可以使用两种结构存储，一种顺序结构，一种链式结构。
  
 顺序存储表示的描述如下：
#define MAX   //二叉树的最大节点数
typedef elemtype SqBitree[MAX]    //0号元素存放根结点
SqBiTree bt;
   
 链式存储结构表示的描述如下：
用链表来表示一颗二叉树，即用链指针来指示元素的逻辑关系。
链表中的每一个结点有三个域组成，除了数据域外，还有两个指针域，分别用来给出该结点的左孩子和右孩子所在的链结点的存储地址。
typedef struct node
{  elemtype data;
     struct node *lchild,*rchild;
}  BiTNode,*BiTree;
将BiTree定义为指向二叉链表结点结构的指针类型. 
5.二叉树的遍历
（1）先序遍历
 根节点——>左子树——>右子树
Void PreOrder(Bitree bt)
{
 //先序遍历二叉树bt
If(bt==NULL) return;
Visite(bt->data);
PreOrder(bt->lchild);
PreOrder(bt->rchild);
}
(2)中序遍历
左子树——>根节点——>右子树
Void InOrder(Bitree bt)
{
 //先序遍历二叉树bt
If(bt==NULL) return;
InOrder(bt->lchild);
Visite(bt->data);
InOrder(bt->rchild);
}
（3）后序遍历
根节点——>左子树——>右子树
Void PostOrder(Bitree bt)
{
 //先序遍历二叉树bt
If(bt==NULL) return;
PostOrder(bt->lchild);
PostOrder(bt->rchild);
Visite(bt->data);

}
6.C++类和对象
C++ 是一门面向对象的编程语言，理解 C++，首先要理解类（Class）和对象（Object）这两个概念。

C++ 中的类（Class）可以看做C语言中结构体（Struct）的升级版。结构体是一种构造类型，可以包含若干成员变量，每个成员变量的类型可以不同；可以通过结构体来定义结构体变量，每个变量拥有相同的性质。例如：
#include <stdio.h>
//定义结构体 Student
struct Student
{
    //结构体包含的成员变量
    char *name;
    int age;
    float score;
};
//显示结构体的成员变量
void display(struct Student stu)
{
    printf("%s的年龄是 %d，成绩是 %f\n", stu.name, stu.age, stu.score);
}
int main()
{
    struct Student stu1;
    //为结构体的成员变量赋值
    stu1.name = "小明";
    stu1.age = 15;
    stu1.score = 92.5;
    //调用函数
    display(stu1);
    return 0;
}
运行结果：
小明的年龄是 15，成绩是 92.500000

C++ 中的类也是一种构造类型，但是进行了一些扩展，类的成员不但可以是变量，还可以是函数；通过类定义出来的变量也有特定的称呼，叫做“对象”。
例如：
#include <stdio.h>
//通过class关键字类定义类
class Student
{
public:
    //类包含的变量
    char *name;
    int age;
    float score;
    //类包含的函数
void say()
{
        printf("%s的年龄是 %d，成绩是 %f\n", name, age, score);
    }
};
int main()
{
    //通过类来定义变量，即创建对象
    class Student stu1;  //也可以省略关键字class
    //为类的成员变量赋值
    stu1.name = "小明";
    stu1.age = 15;
    stu1.score = 92.5f;
    //调用类的成员函数
    stu1.say();
    return 0;
}
运行结果与上例相同。
C语言中的 struct 只能包含变量，而 C++ 中的 class 除了可以包含变量，还可以包含函数。display() 是用来处理成员变量的函数，在C语言中，我们将它放在了 struct Student 外面，它和成员变量是分离的；而在 C++ 中，我们将它放在了 class Student 内部，使它和成员变量聚集在一起，看起来更像一个整体。

结构体和类都可以看做一种由用户自己定义的复杂数据类型，在C语言中可以通过结构体名来定义变量，在 C++ 中可以通过类名来定义变量。不同的是，通过结构体定义出来的变量还是叫变量，而通过类定义出来的变量有了新的名称，叫做对象（Object）。
