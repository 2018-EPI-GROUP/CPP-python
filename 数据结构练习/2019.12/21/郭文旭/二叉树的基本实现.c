#include "stdio.h"
#include "stdlib.h"
#define MAX 20
typedef char TElemType;
typedef int Status;
typedef struct BiTNode
{
	TElemType data;
	struct BiTNode* lchild, * rchild;
} BiTNode, * BiTree;

//先序创建二叉树
void CreateBiTree(BiTree* T)
{
	char ch;
	ch = getchar(); 
	if (ch == '#')	//#代表空指针
		(*T) = NULL;
	else
	{
		(*T) = (BiTree)malloc(sizeof(BiTNode));	//申请结点
		(*T)->data = ch;	//生成根节点
		CreateBiTree(&(*T)->lchild);	//构建左子树
		CreateBiTree(&(*T)->rchild);	//构建右子树
	}
}

//先序输出二叉树
void PreOrder(BiTree T)
{
	if (T)
	{
		printf("%2c", T->data);
		PreOrder(T->lchild);
		PreOrder(T->rchild);
	}
}

//层次遍历二叉树，从一层开始，每层从左到右遍历
void LevelOrder(BiTree T)
{
	BiTree Queue[MAX], b;
	int front, rear;
	front = rear = 0;
	if (T)
	{
		Queue[rear++] = T;
		while (front != rear)
		{
			b = Queue[front++];
			printf("%2c", b->data);
			if (b->lchild != NULL)
				Queue[rear++] = b->lchild;
			if (b->rchild != NULL)
				Queue[rear++] = b->rchild;
		}
	}
}

//求二叉树的深度
int depth(BiTree T)
{
	int dep1, dep2;
	if (T == NULL)
		return 0;
	else
	{
		dep1 = depth(T->lchild);
		dep2 = depth(T->rchild);
		return dep1 > dep2 ? dep1 + 1 : dep2 + 1;
	}
}

//按树状打印二叉树
int PrintTree(BiTree T,int h)
{
	if(T==NULL)
		return 0;
	PrintTree(T->rchild,h+1);
	for(int i=0;i<=h;i++)
		printf(" ");
	printf("%c\n",T->data);
	PrintTree(T->lchild,h+1);
} 

/*
//按树状打印二叉树
int PrintTree(BiTree T,int h)
{
	if(T==NULL)
		return 0;
	PrintTree(T->rchild,h+1);
	for(int i=0;i<=h;i++)
		printf(" ");
	printf("%c\n",T->data);
	PrintTree(T->lchild,h+1);
} 
*/

//统计二叉树的节点数
int Tg(BiTree T)
{
	int a;
	if(T)
	{
		a++;
		Tg(T->lchild);
		Tg(T->rchild);
	}
	return a-1;
}
 
 //输出二叉树叶子节点
 void Iordtea(BiTree T)
 {
 	if(T)
 	{
 		Iordtea(T->lchild);
 		if(T->lchild==NULL&&T->rchild==NULL)
 			printf("%2c",T->data);
 		Iordtea(T->rchild);
 	}
 } 
/* 查找特定值 */
void SearchData(char cha11, BiTree T)
{
    if (T != NULL)
    {
        if (T->data == cha11)
            printf("查找值存在，值为%c\n", T->data);    
            SearchData(cha11, T->lchild);    //递归查找左子树                
            SearchData(cha11, T->rchild);    //递归查找右子树        
    }
}
int main()
{
	int h=0,jiedian;
	char cha11;
	BiTree T = NULL;
	printf("输入链表以#为结尾\n");
	CreateBiTree(&T);
	printf("\n先寻遍历结果");
	PreOrder(T);
	printf("\n层次遍历结果");
	LevelOrder(T);
	jiedian=Tg(T);
	printf("\n二叉树的节点数为%d",jiedian);
	printf("\n输出二叉树的叶子节点");
	Iordtea(T);
	printf("\n树的深度%d\n",depth(T));
	printf("按树状打印二叉树\n");
	PrintTree(T,h);
	printf("输入要查找的值,回车结束");
	getchar();
	cha11=getchar();
	while(cha11!='\n')
	{	
	printf("输入要查找的值,回车结束");
	SearchData(cha11,T);
	getchar();
	cha11=getchar();
	}
}












