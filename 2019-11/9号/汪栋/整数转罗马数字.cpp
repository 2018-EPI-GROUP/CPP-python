int romanToInt(char * s){
    int len = strlen(s);
    if (len == 0£© return 0;
    int ans = 0;
    
    for(int i=0; i < len; ++i) {
        switch (s[i]) {
            case 'M': ans+=1000;break;
            case 'D': ans+=500;break;  
            case 'C': ans+=100; if(i < (len - 1)) if(s[i+1] == 'M' || s[i+1] == 'D') ans-=200; break;
            case 'L': ans+=50; break;
            case 'X': ans+=10; if(i < (len - 1)) if(s[i+1] == 'L' || s[i+1] == 'C') ans-=20; break;
            case 'V': ans+=5;  break;
            case 'I': ans+=1;  if(i < (len - 1)) if(s[i+1] == 'X' || s[i+1] == 'V') ans-=2; break;
            default: break;

                
        }
    }


}
