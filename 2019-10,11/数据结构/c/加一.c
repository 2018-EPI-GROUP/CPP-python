

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* plusOne(int* digits, int digitsSize, int* returnSize){
    int *a;
    a = (int*)malloc(sizeof(int)*(digitsSize+1));
    int i;
    for(i = 0;i < digitsSize+1;i++){
        *(a+i) = 0;
    }
    *(digits+digitsSize-1) += 1;
    for(i = digitsSize-1;i >= 0;i--){
        if(*(digits+i) + *(a+i+1)==10){
            *(a+i+1) = 0;
            *(a+i) += 1;
        }
        else{
            *(a+i+1) += *(digits+i);
        }
    }
    *returnSize = *(a) == 0?digitsSize:digitsSize+1;
    return *(a) == 0?a+1:a;
}

