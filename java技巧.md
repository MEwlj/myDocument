# 1.字符(串)类型对int类型的转换

- Integer.parseInt(String str)

  注意：该方法只对字符串（双引号）进行操作，对字符(单引号)操作会报错。

- Integer.valueOf(String str)

  注意：该方法对单个字符进行操作的时候，不会报错，返回的是字符的ASCII码。该方法也可对字符串进行操作，返回字符串数字对应的整型。

- (int)(a-'0')

  如果a是一个字符数字，也可采用这种办法进行转换。

ps:上述所有的都区分了字符数字和字符串数字，其核心区别在于单引号还是双引号，如果是单个数字，加单引号就是字符，双引号就是字符串。

 # 2. stack.peek()和stack.pop()

- ```
  peek()函数返回栈顶的元素，但不弹出该栈顶元素。
  ```

- ```
  pop()函数返回栈顶的元素，并且将该栈顶元素出栈。
  ```

[csdn](https://blog.csdn.net/EahanZhang/article/details/80755271?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link)



# 3.String和StringBuffer和StringBuilder

- String是不可变字符串，StringBuffer和StringBuilder是可变字符串

StringBuilder有一个append的方法，类似于python的列表

```
class Solution {
    public String addStrings(String num1, String num2) {
        int len1 = num1.length()-1,len2 = num2.length()-1,carry=0;
        StringBuilder res = new StringBuilder("");
        while(len1>=0 || len2 >=0){
            int i = len1 >= 0?num1.charAt(len1)-'0':0;
            int j = len2 >= 0?num2.charAt(len2)-'0':0;
            int sum = i+j+carry;
            carry=sum/10;
            int tmp=sum%10;
            res.append(tmp);
            len1--;len2--;
        }
        if(carry == 1)
            res.append(1);
        return res.reverse().toString();
    }
}
```

力扣415题，两个字符串的数字相加。

