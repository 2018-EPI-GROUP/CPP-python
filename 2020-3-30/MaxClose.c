#include<stdio.h>
#include<time.h>
#include<stdlib.h>

#define InfiniteMax 999
#define InfiniteMin -1

/*利用桶的思想寻找最大相邻数之差 O(N)*/

int getMax(int *A, int N) {
	int Ishave[10];
	int Max[10];
	int Min[10];
	int i;
	for(i=0; i<10; i++) {
		Ishave[i] = 0;
		Max[i] = InfiniteMin;
		Min[i] = InfiniteMax;
	}
	int cur;
	for(i=0; i<N; i++) {
		cur = (A[i]/10)%10;
		Max[cur] =  Max[cur] < A[i]? A[i] : Max[cur];
		Min[cur] =  Min[cur] > A[i]? A[i] : Min[cur];
		Ishave[cur] = Max[cur]==InfiniteMin && Min[cur]==InfiniteMax ?
					  0 : 1;
	}
	
//	for(i=0; i<10; i++)
//	 	printf("%d\t", Ishave[i]);
//	printf("\n");
//	
//	for(i=0; i<10; i++)
//	 	printf("%d\t", Max[i]);	
//	printf("\n");
//	
//	for(i=0; i<10; i++)
//	 	printf("%d\t", Min[i]);
//	printf("\n");
	
	
	
	int ret = 0;
	int j;
	for(i=1; i<10; i++) {
		if(!Ishave[i]) continue;
		//printf("%d %d\n", i, j);
		j = i-1;
		while(j>=0) {
			if(Ishave[j]){
				ret = ret > Min[i]-Max[j]? ret : Min[i]-Max[j];
				printf("%d %d %d\n", i, j, ret);
				break;
			}
			else
				j--;
		} 	
	}
	return ret;
}
/*
int main() {
	int N = 5;
	int A[5] = {12, 11, 15, 30, 9};
	int Max = getMax(A, N);
	printf("%d", Max);	
}
*/

int main() {
	srand(time(NULL));
	int N = rand()%20+1;
	int A[N];
	int i;
	for(i=0; i<N; i++) {
		A[i] = rand()%100;
	}
	for(i=0; i<N; i++)
	 	printf("%d ", A[i]);
	printf("\n");
	
	int Max = getMax(A, N);
	printf("%d", Max);	
}
