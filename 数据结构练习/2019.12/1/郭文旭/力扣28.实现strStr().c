int strStr(char * haystack, char * needle){
    // needle 是空字符串，认为所有字符串都包含空字符串，索引为 0
    if (*needle == 0) return 0;
    // haystack 是空字符串，认为它不包含任何字符串（空字符串除外）
    if (*haystack == 0) return -1;
	int i = -1, j = -1;
	int d = 0, k = -1, s = 0;
	while (needle[s] != '\0')
		s++;
    int next[s+1];
    next[0] = -1;
	while (needle[d] != '\0')
	{
		if (k == -1 || needle[d] == needle[k])
		{
			d++;
			k++;
			next[d] = k;
		}
		else k = next[k];
	}
	while (haystack[i+1] != '\0')
	{
		if(j==-1)
		{
			i++;
			j++;
		}
 			else if (haystack[i] == needle[j] )
			{
				i++;
				j++;
				if (haystack[i + 1] == '\0')
					if (needle[j] == haystack[i] && needle[j + 1] == '\0')
						return i - s + 1;
			}
				else j = next[j];
		if (haystack[i + 1] == '\0')
			if (needle[j] == haystack[i] && needle[j + 1] == '\0')
				return i - s + 1;
		if(j!=-1)
			if (needle[j] == '\0')
				break;
	}
	if (needle[j] == '\0')
		return i - s ;
	else
		return -1;
}
