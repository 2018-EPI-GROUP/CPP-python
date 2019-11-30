#include<stdio.h>
#include<stdlib.h>

#pragma once
#define max 100000000
struct dl {
	int a[max];
	int f, r;
};
int void_dl(struct dl *obj) {
	return obj->f == -1;
}
void in(struct dl *obj, int x) {
	if (void_dl(obj)) {
		obj->f = 0;
		obj->r = 0;
		obj->a[obj->r] = x;
		return;
	}
	obj->r = (obj->r + 1) % max;
	obj->a[obj->r] = x;
	return;
}
int out(struct dl*obj) {
	if (void_dl(obj)) return 0;
	int y;
	if (obj->f == obj->r) {
		y = obj->f;
		obj->f = -1;
		obj->r = -1;
		return obj->a[y];
	}
	y = obj->f;
	obj->f = (obj->f + 1) % max;
	return obj->a[y];
}
int ss(struct dl*obj, int n) {
	int i, j, x, sum = 0;
	int s_f, s_r;
	s_f = obj->f;
	s_r = obj->r;
	for (i = 0; i < (s_r - s_f + 1 + max) % max; i++) {
		x = out(obj);
		for (j = 1; j*j + x <= n; j++) {
			if (x + j * j == n) 
				return 0;
			in(obj, x + j * j);
		}
	}
	return 1;
}


int numSquares(int n) {
	int x;
	struct dl *obj;
	obj = (struct dl*)malloc(sizeof(struct dl));
	obj->f = -1; obj->r = -1;
	in(obj, 0);
	x = 0;
	while (ss(obj, n)) {
		x++;
	}
	return ++x;
}



int main()
{
	int n = 74;
	int x;
	x = numSquares(n);
	printf("%d", x);
	return 0;
}