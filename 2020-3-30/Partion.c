#include<stdio.h>
#include<stdlib.h>
#include<time.h>

/*ºÉÀ¼¹úÆì*/ 

void Swap(int *A, int *B) {
	int tmp = *A;
	*A = *B;
	*B = tmp;
}

void Partion(int *A, int L, int R) {
	if(R-L+1 <= 2)
		return;
	int i;
	int pivot = A[L];
	int cur, Left, Right;
	Left  = cur = L;
	Right = R;
	L--, R++;
//	Swap(&A[0], &A[L+(R-L+1)>>2]);
	while(cur < R) {
		if(A[cur]<pivot)
			Swap(&A[++L], &A[cur++]);
		else if(A[cur]>pivot)
			Swap(&A[--R], &A[cur]);
		else
			cur++;
	}
	//Partion(A, Left, L);
//	Partion(A, R, Right);
}



int main() {
	srand(time(NULL));
	int N = rand()%20+1;
	int A[N];
	int i;
	for(i=0; i<N; i++) {
		A[i] = rand()%10;
	}
	for(i=0; i<N; i++) {
		printf("%d ",A[i]);
	}
	printf("\n");
	
	Partion(A, 0, N-1);
	
	for(i=0; i<N; i++) {
		printf("%d ",A[i]);
	}
	//print();
}
