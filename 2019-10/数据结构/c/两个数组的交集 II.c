

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* intersect(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
    int i,j,n,x,flog=0;
    int min = nums1Size<nums2Size?nums1Size:nums2Size+1;
    int *nums3;
    n = 0;
    nums3 = (int*)malloc(sizeof(int)*min);
    int nums4[nums2Size+1];
    for(i = 0;i < nums2Size;i++){
        nums4[i]  = 0;
    }
    for(i = 0;i < nums1Size;i++){
        flog = 0;
        for(j = 0;j < nums2Size;j++){
            if(nums1[i] == nums2[j]){
                if(aaa(nums3,nums1[i],n)==-1){
                    *(nums3+n) = nums1[i];
                    nums4[j] = 1;
                    flog = 1;
                    n++;
                }
                else if(flog == 0){
                    if(nums4[j] == 0){
                        *(nums3+n) = nums1[i];
                        nums4[j] = 1;
                        flog = 1;
                        n++;
                    }
                }
            }
        }
    }
    for(i = 0;i < n;i++){
        printf("%d ",nums3[i]);
    }
    *returnSize = n;
   return nums3;
}
int aaa(int*a,int x,int n){
    int i;
    for(i = 0;i < n;i++){
        if(a[i] == x){
            return i;
        }
    }
    return -1;
}
