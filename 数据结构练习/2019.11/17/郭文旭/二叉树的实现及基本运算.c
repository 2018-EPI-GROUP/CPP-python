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

int main()
{
	int h=0;
	BiTree T = NULL;
	printf("输入链表以#为结尾\n");
	CreateBiTree(&T);
	printf("\n先寻遍历结果");
	PreOrder(T);
	printf("\n层次遍历结果");
	LevelOrder(T);
	printf("\n树的深度%d\n",depth(T));
	printf("按树状打印二叉树\n");
	PrintTree(T,h);
}












