#include<stdio.h>
#include<stdlib.h>

#pragma once
#define MAX 10000
struct duilie {
	char* a[MAX];
	int f, r;
};
int void_dl(struct duilie *obj) {
	return (obj->f == -1);
}
void in(struct duilie *obj, char *s) {
	char *y;
	y = (char*)malloc(sizeof(char)*5);
	int i;
	for (i = 0; i < 5; i++) {
		*(y + i) = *(s + i);
	}
	if (void_dl(obj)) {
		obj->f = 0;
		obj->r = 0;
		obj->a[obj->r] = y;
		return;
	}
	obj->r = (obj->r + 1) % MAX;
	obj->a[obj->r] = y;
	return;
}
struct duilie* out(struct duilie *obj) {
	int x;
	if (void_dl(obj)) return NULL;
	if (obj->f == obj->r) {
		x = obj->f;
		obj->f = -1;
		obj->r = -1;
		return obj->a[x];
	}
	x = obj->f;
	obj->f = (obj->f + 1) % MAX;
	return obj->a[x];
}
int strpd(char *s1, char *s2) {
	int i;
	for(i = 0;i < 4;i++) {
		if (*s1 == *s2) {
			s1++;
			s2++;
		}
		else return 0;
	}
	return 1;
}
int sousuo(struct duilie*obj, char ** deadends, int deadendsSize, char * target,int pc[10][10][10][10]) {
	int s_f, s_r;
	s_f = obj->f;
	s_r = obj->r;
	if (s_f == -1) return -1;
	int i, j;
	char *s;
	int flag = 1;
	for (i = 0; i<(s_r-s_f+MAX+1)%MAX ; i++) {
		flag = 1;
		s = out(obj);
		if (s == NULL) 
			return -1;
		if (strpd(s, target)) 
			return 0;
		1;
		for (j = 0; j < deadendsSize; j++) {
			if (strpd(*(deadends + j), s)) {
				flag = 0;
				break;
			}
		}
		if (flag) {
			for (j = 0; j < 4; j++) {
				s[j] = '0' + (s[j] - '0' + 1) % 10;
				if (!pc[s[0] - '0'][s[1] - '0'][s[2] - '0'][s[3] - '0']) {
					pc[s[0] - '0'][s[1] - '0'][s[2] - '0'][s[3] - '0'] = 1;
					in(obj, s);
				}
				s[j] = '0' + (s[j] - '0' - 1 + 10) % 10;
			}
			for (j = 0; j <4; j++) {
				s[j] = '0' + (s[j] - '0' - 1 + 10) % 10;
				if (!pc[s[0] - '0'][s[1] - '0'][s[2] - '0'][s[3] - '0']) {
					pc[s[0] - '0'][s[1] - '0'][s[2] - '0'][s[3] - '0'] = 1;
					in(obj, s);
				}
				s[j] = '0' + (s[j] - '0' + 1) % 10;
			}
		}
	}
	return 1;
}
int openLock(char ** deadends, int deadendsSize, char * target) {
	int x;
	struct duilie *obj;
	obj = (struct duilie *)malloc(sizeof(struct duilie));
	obj->f = -1; obj->r = -1;
	char *cs; int i;
	cs = (char*)malloc(sizeof(char) * 5);
	for (i = 0; i < 4; i++) { *(cs + i) = '0'; }
	*(cs + i) = '\0';
	in(obj, cs);
	x = 0;
	int y;
	int pc[10][10][10][10] = { 0 };
	pc[0][0][0][0] = 1;
	while (y = sousuo(obj, deadends, deadendsSize, target,pc)) {
		x++;
		if (x > 10000) return -1;
		if (y == -1) {
			return -1;
		}
	}
	return x;
}



#define X 8
int main()
{
	char a[X][5] = { "8887\0","8889\0","8878\0","8898\0","8788\0","8988\0","7888\0","9888\0" };
	char b[5] = { "8888\0" };
	int c = X ,x;
	char ** y;
	int i, j, n;
	y = (char**)malloc(sizeof(char*) * X);
	for (i = 0; i < X; i++) {
		*(y + i) = (char*)malloc(sizeof(char) * 5);
		for (j = 0; j < 5; j++) {
			*(*(y + i) + j) = a[i][j];
		}
	}
	x = openLock(y, c, b);
	
	printf("%d", x);
	return 0;
}
