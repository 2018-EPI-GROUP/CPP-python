
#define a(x,y) x^=y;y^=x;x^=y
void rotate(int** matrix, int matrixSize, int* matrixColSize){
    int i,j;
    for(i = 0;i < matrixSize/2;i++){
        for(j = i;j < matrixSize-i-1;j++){
            a(*(*(matrix+i)+j),*(*(matrix+j)+matrixSize-1-i));
            a(*(*(matrix+i)+j),*(*(matrix+matrixSize-1-i)+matrixSize-1-j));
            a(*(*(matrix+i)+j),*(*(matrix+matrixSize-1-j)+i));
        }
    }
}

