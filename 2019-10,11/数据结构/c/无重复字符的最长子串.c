

int lengthOfLongestSubstring(char * s){
    char *c;
    if(*s == '\0'){
        return 0;
    }
    int csize;
    c = s;
    csize = 0;
    int i,x,y = 1;
    x = 0;
        
            while(*(c+csize) != '\0'){
                csize++;
                while(x<csize){
                    if(*(c+csize) == *(c+x) || *(c+csize) == '\0'){
                        if(csize > y){
                            y = csize;
                        }
                        c = c+x+1;
                        csize = 0;
                        x = 0;
                        break;
                    }
                    x++;
                }
                //csize++;
                x = 0;
            }
            
        return y;
    
}

