

void moveZeroes(int* nums, int numsSize){
    int *q,*p,*r;
    q = nums;
    p = nums;
    r = nums;
    int i,j,flog;
    flog = 0;
    for(i = 0;i < numsSize;i++){
        if(*q==0){
            if(!flog){
                flog = 1;
                r = q;
                p = q;
                //printf("%d\n",i);
                
            }
            else{
                p++;
            }
            //printf("2 %d\n",i);
            q++;
        }
        else if(flog){
            *r^=*q;
            *q^=*r;
            *r^=*q;
            r++;
            p++;
            q++;
            //printf("3 %d\n",i);
        }
        else{
            q++;
        }
    }
}

