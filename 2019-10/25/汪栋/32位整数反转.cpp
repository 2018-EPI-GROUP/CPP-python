#include<stdio.h>
int reverse(int x){
        long y = 0;
        while (x != 0) {
            y = y * 10 + x % 10;
            x /= 10;
        }
    if(-2147483648>y || y>2147483647)
       return 0;
    return (int) y;
    
    
    }
int main()
{
	int x;
	scanf("%d",&x);
	printf("%d",reverse(x));
	return 0;
}
