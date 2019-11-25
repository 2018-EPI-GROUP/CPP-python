int isValid(char* s) {
	int top = 0, i = 0,j = 0;
	char* p;
    p = (char *)malloc(strlen(s) + 1);
    while (s[j])
		j++;
	if (j % 2 != 0)
		return false;
	if (s == NULL || s[0] == '\0')
		return true;
	for (i = 0; s[i]; i++)
		if (s[i] == '(' || s[i] == '[' || s[i] == '{')
		{
			p[top] = s[i];
			top++;
		}
        else if ((--top) < 0)
            return false;
		else if (s[i] == ')' && p[top] != '(')
			return false;
		else if (s[i] == ']' && p[top] != '[')
			return false;
		else if (s[i] == '}' && p[top] != '{')
			return false;
	if (s[0] == ')' || s[0] == '}' || s[0] == ']')
		return false;
	else if(top!=0)
		return false;
	else
		return true;
}
