int Read(char s)
{
	switch (s)
	{
	case 'I' :return 1;
	case 'V' :return 5;
	case 'X' :return 10;
	case 'L' :return 50;
	case 'C' :return 100;
	case 'D' :return 500;
	case 'M' :return 1000;
	}
    return 0;
}

int romanToInt(char* s) {
	int Rin, Lin, i = 0, j = 1, sum = 0;
	char Rread, Lread;
	while (s[i] != '\0')
	{
		Rread = s[i];
        if (s[j] == '\0')
		{
			Rin = Read(Rread);
			return Rin;
		}
		Lread = s[j];
		Rin = Read(Rread);
		Lin = Read(Lread);
		if (Rin >= Lin)
			sum = Rin + sum;
		else
			sum = sum - Rin;
		i++;
		j++;
		if (s[j] == '\0')
		{
			sum = sum + Lin;
			return sum;
		}
	}
    return 0;
}
