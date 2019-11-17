

bool isValidSudoku(char** board, int boardSize, int* boardColSize){
    int i,j,n,a[10][10],b[10],c[10][10];
    for(i = 0;i < 10;i++){
        b[i] = 0;
        for(j = 0;j < 10;j++){
            a[i][j] = 0;
            c[i][j] = 0;
        }
    }
    for(i = 1;i <= 9;i++){
        for(j = 1;j <=9;j++){
            if(board[i-1][j-1] == '.'){
                continue;
            }
            else{
                a[j][(int)(board[i-1][j-1] - '0')]++;
                b[(int)(board[i-1][j-1] - '0')]++;
                c[((i-1)/3)*3+(j-1)/3+1][(int)(board[i-1][j-1] - '0')]++;
                //printf("%d ",(int)(board[i-1][j-1] - '0'));
                if(a[j][(int)(board[i-1][j-1] - '0')]>1||b[(int)(board[i-1][j-1] - '0')]>1||c[((i-1)/3)*3+(j-1)/3+1][(int)(board[i-1][j-1] - '0')]>1){
                    //printf("%d %d %d ",a[i][(int)(board[i-1][j-1] - '0')],b[(int)(board[i-1][j-1] - '0')],c[(i/3)*3+j/3+1][(int)(board[i-1][j-1] - '0')]);
                   //printf("%d ",(int)(board[i-1][j-1] - '0'));
                    //printf("%d %d ",i,j);
                    //printf("%d",(i/3)*3+j/3+1);
                    return false;
                }
            }
        }
        for(n = 0;n < 10;n++){
                b[n] = 0;
            }
    }
    return true;
}

