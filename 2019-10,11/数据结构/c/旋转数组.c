#include<stdio.h>
#include<malloc.h>

void rotate(int* nums, int numsSize, int k){
    int i,a;
    if(k == 0||numsSize == 1) return;
    if(k > numsSize){
        k = k%numsSize;
    }
    int b[(numsSize/k)*k+k];
    for(i = 0;i <(numsSize/k)*k ;i++){
        b[i] = nums[i];
    }
    for(i = 0;i < k;i++){
        b[(numsSize/k)*k+i] = nums[numsSize-k+i];
    }
    
    for(i = 0;i < k;i++){
        for(a = i+k;a < (numsSize/k)*k+k;a += k){
            b[i]^=b[a];
            b[a]^=b[i];
            b[i]^=b[a];
        }
        
    }
    for(i = 0;i < numsSize;i++){
        nums[i] = b[i];
    }
}

