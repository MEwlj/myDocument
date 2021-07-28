# 1.单列集合框架结构

本文选自尚硅谷视频

|----Collection接口：单列集合，用来存储一个一个的对象

*          |----List接口：存储序的、可重复的数据。  -->“动态”数组
*              |----ArrayList、LinkedList、Vector
*
*          |----Set接口：存储无序的、不可重复的数据   -->高中讲的“集合”
*              |----HashSet、LinkedHashSet、TreeSet

# 2.常用方法

## 2.1 contains(Object obj) 

1. contains(Object obj)：判断当前集合中是否包含obj
2. 返回值为boolean类型
3. 该方法运行的时候是将contains内的内容依次与集合中的元素进行比较，且如果obj是引用数据类型的话，比较的时候调用的是obj对象所在类的equals方法。

ps：

- equals是判断两个变量或者实例指向同一个内存空间的值是不是相同

- 而==是判断两个变量或者实例是不是指向同一个内存空间

- <u>**如果obj是自己定义的一个类，需要重写equals方法，因为默认的equals方法是调用的‘==’来进行判断的。**</u>

- 重写后的代码如下：（选自尚硅谷视频）

  ![image-20210726154123528](./picture/java集合\image-20210726154123528.png)

4. List和Set集合都可调用contains方法。

## 2.2 containsAll(Collection coll1)

1. containsAll(Collection coll1) 判断形参coll1中的所有元素是否都存在于当前集合中。
2. 形参必须为一个集合，可以采用**Arrays.asList**方法快速建立集合。
3. 返回值为boolean类型。

代码演示：

```java
    @Test
    public void test1(){
        Collection coll = new ArrayList();
        coll.add(123);
        coll.add(456);
//        Person p = new Person("Jerry",20);
//        coll.add(p);
        coll.add(new Person("Jerry",20));
        coll.add(new String("Tom"));
        coll.add(false);
        //1.contains(Object obj):判断当前集合中是否包含obj
        //我们在判断时会调用obj对象所在类的equals()。
        boolean contains = coll.contains(123);
        System.out.println(contains);
        System.out.println(coll.contains(new String("Tom")));//true
//        System.out.println(coll.contains(p));//true
        System.out.println(coll.contains(new Person("Jerry",20)));//false -->true（equals重写前-->重写后）

        //2.containsAll(Collection coll1):判断形参coll1中的所有元素是否都存在于当前集合中。
        Collection coll1 = Arrays.asList(123,4567);
        System.out.println(coll.containsAll(coll1));
    }
```

## 2.3 remove(Object obj)

1. 当前集合删除obj元素
2. 返回值为boolean类型

## 2.4 removeAll(Collection coll1)

1. removeAll(Collection coll1) 从当前集合中移除coll1中所有的元素
2. 实现了数学上的差集概念。
3. 调用该方法是修改当前集合的意思，无返回值。



![image-20210726155858790](./picture/java集合\image-20210726155858790.png)

代码演示

```java
    @Test
    public void test2(){
        //3.remove(Object obj):从当前集合中移除obj元素。
        Collection coll = new ArrayList();
        coll.add(123);
        coll.add(456);
        coll.add(new Person("Jerry",20));
        coll.add(new String("Tom"));
        coll.add(false);

        coll.remove(1234);
        System.out.println(coll);

        coll.remove(new Person("Jerry",20));
        System.out.println(coll);

        //4. removeAll(Collection coll1):差集：从当前集合中移除coll1中所有的元素。
        Collection coll1 = Arrays.asList(123,4567);//仅移除了coll集合中的123
        coll.removeAll(coll1);
        System.out.println(coll);
```



## 2.5 retainAll(Collection coll1)

1. 获取当前集合和coll1集合的交集。
2. 无返回值（或者说返回给当前集合）



## 2.6 equals(Object obj)

1. equals(Object obj) 方法内的obj对象可以是集合。
2. 如果obj对象是List集合的话，则在调用equals方法的时候，也会进行集合元素顺序的比较

代码演示

```java
    @Test
    public void test3(){
        Collection coll = new ArrayList();
        coll.add(123);
        coll.add(456);
        coll.add(new Person("Jerry",20));
        coll.add(new String("Tom"));
        coll.add(false);

        //5.retainAll(Collection coll1):交集：获取当前集合和coll1集合的交集，并返回给当前集合
//        Collection coll1 = Arrays.asList(123,456,789);
//        coll.retainAll(coll1);
//        System.out.println(coll);

        //6.equals(Object obj):要想返回true，需要当前集合和形参集合的元素都相同。
        Collection coll1 = new ArrayList();
        coll1.add(456);
        coll1.add(123);
        coll1.add(new Person("Jerry",20));
        coll1.add(new String("Tom"));
        coll1.add(false);

        System.out.println(coll.equals(coll1));//返回为false，因为两个集合头两个元素的顺序不一样，且ArrayList集合是有序的，而HashSet集合是无序的。
```



## 2.7 hashCode()

1. 该方法在object对象内部，故所有类都有该方法，可以用于确定对象的存储位置。
2. 返回当前对象的hash值。

代码演示

```java
System.out.println(coll.hashCode());
```



## 2.8 toArray()

1. 集合转换为数组 
2. coll.toArray()

代码演示

```java
    @Test
    public void test4(){
        Collection coll = new ArrayList();
        coll.add(123);
        coll.add(456);
        coll.add(new Person("Jerry",20));
        coll.add(new String("Tom"));
        coll.add(false);

        //7.hashCode():返回当前对象的哈希值
        System.out.println(coll.hashCode());

        //8.集合 --->数组：toArray()
        Object[] arr = coll.toArray();
        for(int i = 0;i < arr.length;i++){
            System.out.println(arr[i]);
        }
```



ps：此时输出的array的类型并不是固定的，因为是一个Object类型的数组

![image-20210726164046114](./picture/java集合\image-20210726164046114.png)

 

## 2.9  Arrays.asList()

1. 数组转化为集合

ps：

- 假如数组是一个整形数组，在转化的时候必须要是Integer包装类型，而不能是int类型。

![image-20210726164541257](./picture/java集合\image-20210726164541257.png)

- 改写成

```
List arr1 = Arrays.asList(123,456);
```

或者

```
List arr1 = Arrays.asList(new Integer[]{123,456});
```

ps：在集合后面的<>似乎可以省略掉。



代码演示

```java
    @Test
    public void test4(){
        Collection coll = new ArrayList();
        coll.add(123);
        coll.add(456);
        coll.add(new Person("Jerry",20));
        coll.add(new String("Tom"));
        coll.add(false);

        //7.hashCode():返回当前对象的哈希值
        System.out.println(coll.hashCode());

        //8.集合 --->数组：toArray()
        Object[] arr = coll.toArray();
        for(int i = 0;i < arr.length;i++){
            System.out.println(arr[i]);
        }

        //拓展：数组 --->集合:调用Arrays类的静态方法asList()
        List<String> list = Arrays.asList(new String[]{"AA", "BB", "CC"});
        System.out.println(list);

        List arr1 = Arrays.asList(new int[]{123, 456});
        System.out.println(arr1.size());//1

        List arr2 = Arrays.asList(new Integer[]{123, 456});
        System.out.println(arr2.size());//2
```

## 2.10 其他常用方法

![image-20210728111709934](./picture/java集合\image-20210728111709934.png)

增：add(Object obj)
删：remove(int index) / remove(Object obj)
改：set(int index, Object ele)
查：get(int index)
插：add(int index, Object ele)
长度：size()
遍历：① Iterator迭代器方式
     ② 增强for循环
     ③ 普通的循环
