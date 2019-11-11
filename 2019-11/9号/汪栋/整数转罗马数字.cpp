int romanToInt(char* s) {
    int len,lenn,k=0,h=0,m=0;
    len=strlen(s);lenn=len;
    int count=0;
    char a[7]={'I','V','X','L','C','D','M'}; 
    int b[7]={1,5,10,50,100,500,1000};
  for(int i=0;i<len;i++)  
  {   lenn--;
      for(int j=0;j<7;j++) {  if (s[i]==a[j])  {k=j;break;} }
      for(int j=0;j<7;j++) { if (s[i+1]==a[j])  {h=j;break;}}
      count+=b[k]; 
  if (k==0 && (h==1||h==2)) {count+=-2*b[0]+b[h];i++;lenn--;}
  if (k==2 && (h==3||h==4)) {count+=-2*b[2]+b[h];i++;lenn--;}
  if (k==4 && (h==5||h==6)) {count+=-2*b[4]+b[h];i++;lenn--;}

   if (lenn==1)  { for(int j=0;j<7;j++) { if (s[i+1]==a[j])  {h=j;break;}} count+=b[h];  break;}
   if (lenn==0)  break;
    }

 return count;  
}

