

bool containsDuplicate(int* nums, int numsSize){
    if(numsSize == 0 || numsSize ==1) return false;
    int i;
    int max = nums[0],min = nums[0];
    for(i = 0;i < numsSize;i++){
        if(nums[i] > max){
            max = nums[i];
        }
        if(nums[i] < min){
            min = nums[i];
        }
    }
    int a[max - min + 1];
    for(i = 0;i < max-min+1;i++){
        a[i] = 0;
    }
    for(i = 0;i < numsSize;i++){
        if(a[nums[i]-min] != 0){
            
            return true;
        }
        a[nums[i]-min] = 1;
    }
    return false;
    
}

