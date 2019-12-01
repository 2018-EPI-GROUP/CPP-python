# c++类和对象的深度学习

1.构造函数是一个特殊的成员函数，名字与类名相同,创建类类型对象时由编译器自动调用，保证每 个数据成员都有 一个合适的初始值，并且在对象的生命周期内只调用一次。

```c++
/*class Date
{
public:
	void SetDate(int year, int month, int day)
	{
		_year = year;
		_month = month;
		_day = day;		
	}
	void Display()
	{
		cout << _year << "-" << _month << "-" << _day << endl;
	}
private:
	int _year;
	int _month;
	int _day;
};
int main()
{
	Date d1, d2;
	d1.SetDate(2019, 11, 30);
	d1.Display();
	d2.SetDate(2019, 11, 29);
	d2.Display();
	return 0;
}
```

2.2.2 特性 
构造函数是特殊的成员函数，需要注意的是，构造函数的虽然名称叫构造，但是需要注意的是构 造函数的主要任务并不是开空间创建对象，而是初始化对象。
其特征如下： 
1. 函数名与类名相同。 

2. 无返回值。 

3. . 对象实例化时编译器自动调用对应的构造函数。

4.   构造函数可以重载。 

5. 如果类中没有显式定义构造函数，则C++编译器会自动生成一个无参的默认构造函数，一旦 用户显式定义编译器将不再生成。

6. ```c++
   class Date
   {
   public:
   	//如果用户定义了构造函数，则编译器不再生成
   	//Date(int year, int month, int day)
   	//{
   		//_year = year;
   		//_month = month;
   		//_day = day;
   	//}
   private:
   	int _year;
   	int _momth;
   	int _day;
   };
   void main()
   {
   	Date d;
   }
   ```

   6. 无参的构造函数和全缺省的构造函数都称为默认构造函数，并且默认构造函数只能有一个。 注意：无参构造函数、全缺省构造函数、我们没写编译器默认生成的构造函数，都可以认为 是默认成员函数。

   7. ：C++把类型分成内置类型(基本类型)和自定义类型。内置类型就是语法已经定义好的类型： 如int/char...，自定义类型就是我们使用class/struct/union自己定义的类型，看看下面的程序， 就会发现编译器生成默认的构造函数会对自定类型成员_t调用的它的默认成员函数

      # 析构函数

      3.析构函数 
      3.1 概念 
      前面通过构造函数的学习，我们知道一个对象时怎么来的，那一个对象又是怎么没呢的？
      析构函数：与构造函数功能相反，析构函数不是完成对象的销毁，局部对象销毁工作是由编译器 完成的。而对象在销毁时会自动调用析构函数，完成类的一些资源清理工作。

      3.2 特性 
      析构函数是特殊的成员函数。
      其特征如下： 

      1. 析构函数名是在类名前加上字符 ~。 

      2. 无参数无返回值。 

      3. 一个类有且只有一个析构函数。若未显式定义，系统会自动生成默认的析构函数。 

      4.  对象生命周期结束时，C++编译系统系统自动调用析构函数。

      5. ```c++
         class SeqList
         {
         public:
         	SeqList(int capacity = 10)
         	{
         		_pdata = (int*)malloc(capacity * sizeof(int));
         		assert(_pdata);
         		_size = 0;
         		_capacity = capacity;
         	}
         	~SeqList()
         	{
         		if (_pdata)
         		{
         			free(_pdata);
         			_pdata = NULL;
         			_size = 0;
         			_capacity = 0;			
         		}
         	}
         private:
         	int* _pdata;
         	size_t _size;
         	size_t _capacity;
         };
         ```

         

   4. 拷贝构造函数 
   5. 构造函数：只有单个形参，该形参是对本类类型对象的引用(一般常用const修饰)，在用已存在的 类类型对象创建新对象时由编译器自动调用。
   6. 若未显示定义，系统生成默认的拷贝构造函数。 默认的拷贝构造函数对象按内存存储按字节 序完成拷贝，这种拷贝我们叫做浅拷贝，或者值拷贝。
   7. 5.赋值运算符重载 
      #####5.1 运算符重载
      C++为了增强代码的可读性引入了运算符重载，运算符重载是具有特殊函数名的函数，也具有其 返回值类型，函数名字以及参数列表，其返回值类型与参数列表与普通的函数类似。
      函数名字为：关键字operator后面接需要重载的运算符符号。
      函数原型：返回值类型 operator操作符(参数列表)
      注意：
          int _month;    int _day; };
      int main() {    Date d1;    // 这里d2调用的默认拷贝构造完成拷贝，d2和d1的值也是一样的。    Date d2(d1);
          return 0; }
      // 这里会发现下面的程序会崩溃掉？这里就需要我们以后讲的深拷贝去解决。 class String { public:    String(const char* str = "jack")    {        _str = (char*)malloc(strlen(str) + 1);        strcpy(_str, str);    }
          ~String()    {        cout << "~String()" << endl;        free(_str);    } private:    char* _str; };
      int main() {    String s1("hello");    String s2(s1); } 
      不能通过连接其他符号来创建新的操作符：比如operator@ 重载操作符必须有一个类类型或者枚举类型的操作数 用于内置类型的操作符，其含义不能改变，例如：内置的整型+，不 能改变其含义 作为类成员的重载函数时，其形参看起来比操作数数目少1成员函数的 操作符有一个默认的形参this，限定为第一个形参 .* 、:: 、sizeof  、?:   、.  注意以上5个运算符不能重载。这个经常在笔试选择题中出现

 赋值运算符主要有四点：
1. 参数类型 

2. 返回值 

3. 检测是否自己给自己赋值 

4. 返回*this 

5. 一个类如果没有显式定义赋值运算符重载，编译器也会生成一个，完成对象按字节序的值拷 贝。

6. ```c++
   class Date
   { 
       public:    
    
       Date(int year = 1900, int month = 1, int day = 1)    
    {        
           _year = year;        
           _month = month;        
           _day = day;    
       } 
       private:    
       int _year;    
       int _month;    
       int _day; 
   };
   int main() 
   {    
       Date d1;    
        Date d2(2018，10， 1);        // 这里d1调用的编译器生成operator=完成拷贝，d2和   d1    的值也是一样的。    
       d1 = d2；
       return 0;
   }
   ```

   

