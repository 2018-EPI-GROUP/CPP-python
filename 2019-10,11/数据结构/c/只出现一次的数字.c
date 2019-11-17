

int singleNumber(int* nums, int numsSize){
    if(numsSize == 1) return nums[0];
    int i,n = nums[0];
    for(i = 1;i < numsSize;i++){
        n ^= *(nums+i);
    }
    return n;
}

