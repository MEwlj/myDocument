# 1.java字符串常用方法

## 1.1 str.toCharArray()

该方法能把字符串转换为字符数组



## 1.2 str.length()

```java
int i = str.length();
```

与数组的length属性不同，数组的length表示属性，而字符串的length()表示方法。



## 1.3 str.isEmpty()

```java
String str = "";
boolean res = str.isEmpty(); //true
```

判断字符串是否为空字符串。



## 1.4 str.trim()

将字符串头尾的空格（空白符）去掉，常常配合isEmpty()使用

```java
String str = "   ";
if((digits.isEmpty())||(digits.trim().isEmpty())){
            return;
}
```



## 1.5 str.charAt(int index)

此方法返回位于字符串的指定索引处的**字符**。该字符串的索引从零开始

```java
String str = "123";
char res = str.charAt(1);// res = '2'
```



## 1.6 str.toCharArray()

将字符串转化为字符数组

```
    public int longestPalindrome(String s) {
        char[] arr = s.toCharArray();
    }
```

