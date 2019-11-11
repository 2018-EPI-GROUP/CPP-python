#include<stdio.h>
bool isPalindrome(int x){
    int k = x;
    long y = 0;
    if(x<0)
        printf("false");
    else
    {
    while(x!=0)
    {
        y = y*10 + x%10;
        x = x/10;
    }
    }
    if(k==y)
    {
    	printf("true");
    	return true;
    }
        
    else
    printf("false");
       return false;

}
int main()
{
	int x;
	scanf("%d",&x);
	isPalindrome(x);

	
	return 0;
} 
