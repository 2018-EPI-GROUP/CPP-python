#include<stdio.h>
#include<stdlib.h>


#pragma once
struct wz {
	int x, y;
};
struct dl {
	struct wz wz[1000];
	int f, r;
}*grid_dl;

int in(struct dl *obj, int grid_x, int grid_y) {
	if (dl_void(obj)) {
		obj->f = 0;
		obj->r = 0;
		obj->wz[0].x = grid_x;
		obj->wz[0].y = grid_y;
		return 1;
	}
	if (obj->r + 1 == obj->f) return 0;
	obj->r = (obj->r + 1) % 1000;
	obj->wz[obj->r].x = grid_x;
	obj->wz[obj->r].y = grid_y;
	return 1;
}
int out(struct dl *obj, int *grid_x, int *grid_y) {
	int val = 1;
	if (dl_void(obj)) return 0;
	if (obj->f == obj->r) {
		*grid_x = obj->wz[obj->f].x;
		*grid_y = obj->wz[obj->f].y;
		obj->f = -1;
		obj->r = -1;
		return val;
	}
	*grid_x = obj->wz[obj->f].x;
	*grid_y = obj->wz[obj->f].y;
	obj->f = (1000 + obj->f + 1) % 1000;
	return val;
}
int dl_void(struct dl *obj) {
	return (obj->f == -1);
}
int gridpd(char** grid, int gridSize, int* gridColSize) {
	if (dl_void(grid_dl)) return 0;
	int grid_x, grid_y;
	out(grid_dl, &grid_x, &grid_y);
	pd(grid, gridSize, gridColSize, grid_x, grid_y);
	return gridpd(grid, gridSize, gridColSize);
}
int pd(char ** grid, int gridSize, int *gridColSize, int grid_x, int grid_y) {
	if (grid_y + 1 < *(gridColSize + grid_x) && *(*(grid + grid_x) + grid_y + 1) == '1') {
		*(*(grid + grid_x) + grid_y + 1) = '0';
		in(grid_dl, grid_x, grid_y + 1);
	}
	if (grid_x + 1 < gridSize&&*(*(grid + grid_x + 1) + grid_y) == '1') {
		*(*(grid + grid_x + 1) + grid_y) = '0';
		in(grid_dl, grid_x + 1, grid_y);
	}
	if (grid_y - 1 >= 0 && *(*(grid + grid_x ) + grid_y - 1) == '1') {
		*(*(grid + grid_x ) + grid_y - 1) = '0';
		in(grid_dl, grid_x , grid_y - 1);
	}
	if (grid_x - 1 >= 0 && *(*(grid + grid_x - 1) + grid_y) == '1') {
		*(*(grid + grid_x - 1) + grid_y) = '0';
		in(grid_dl, grid_x - 1, grid_y);
	}
	return 1;
}

int numIslands(char** grid, int gridSize, int* gridColSize) {

	int i, j, k = 0;
	grid_dl = (struct dl*)malloc(sizeof(struct dl));
	grid_dl->f = -1;
	grid_dl->r = -1;
	for (i = 0; i < gridSize; i++) {
		for (j = 0; j < *(gridColSize + i); j++) {
			if (grid[i][j] == '1') {
				in(grid_dl, i, j);
				*(*(grid + i) + j) = '0';
				gridpd(grid, gridSize, gridColSize);
				k++;
			}
		}
	}
	return k;
}






int main()
{
	char b[3][5] = {
		{'1','0','1','1','1'},
		{'1','0','1','0','1'},
		{'1','1','1','0','1'}
	};
	char **a;
	a = (char**)malloc(sizeof(char*) * 4);
	char *x;
	int i,j;
	for (i = 0; i < 4; i++) {
		x = (char*)malloc(sizeof(char));
		*(a + i) = x;
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 5; j++) {
			*(*(a + i) + j) = b[i][j];
		}
	}
	int c[3] = { 3,3,3 };
	int k;
	k = numIslands(a, 5, c);
	printf("%d", k);
	return 0;
}