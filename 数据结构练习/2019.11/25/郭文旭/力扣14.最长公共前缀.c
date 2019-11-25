char * longestCommonPrefix(char ** strs, int strsSize){
    int i = 0, j = 0;
	if (strsSize == 0)
		return "";
	if (strsSize == 1)
		return strs[0];
	while (1)
	{
		for (i = 0; i < strsSize; ++i)
		{
			if (strs[0][j] != strs[i][j] || (!strs[i][j]))
			{
				strs[i][j] = '\0';
				return strs[i];
			}
		}
		j++;
	}

}
