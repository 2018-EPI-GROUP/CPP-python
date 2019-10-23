#include<stdio.h>

int maxProfit(int* prices, int pricesSize){
    if(pricesSize == 0||pricesSize == 1){
        return 0;
    }
    int a[pricesSize-1];
    int i;
    for(i = 0;i < pricesSize-1;i++){
        a[i] = prices[i+1] - prices[i];
    }
    int sum = 0,ssum = 0;
    for(i = 0;i < pricesSize-1;i++){
        if(a[i] <= 0){
            ssum += sum;
            sum = 0;
        }
        else{
            sum += a[i];
        }
    }
    ssum += sum;
    return ssum;
}

