# 1.单列集合框架结构

本文选自尚硅谷视频

|----Collection接口：单列集合，用来存储一个一个的对象

*          |----List接口：存储序的、可重复的数据。  -->“动态”数组
*              |----ArrayList、LinkedList、Vector
*
*          |----Set接口：存储无序的、不可重复的数据   -->高中讲的“集合”
*              |----HashSet、LinkedHashSet、TreeSet

# 2.Link常用方法

常用的15个抽象方法（实现自Collection的抽象方法），最常用的以下5个。这儿不包括迭代器方法。

1. add(Object e):元素e添加到集合coll内
2. size():获取添加元素的个数
3. addAll():将coll1集合中的元素添加到当前的集合中

```java
Collection coll1 = new ArrayList();
coll1.add(456);
coll1.add("CC");
coll.addAll(coll1);
```

4. isEmpty():判断当前集合是否为空（观察size()是否为0）
5. clear():清空集合元素，并不是给coll赋值为null，而是直接清空集合元素。

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



ps：此时输出的array的类型并不是固定的，因为是一个Object类型的数组，应该是用到了多态的思想。

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

## 2.10 其他常用方法（List特有，Set没有）

![image-20210728111709934](./picture/java集合\image-20210728111709934.png)

增：add(Object obj)，返回布尔类型
删：remove(int index) / remove(Object obj)  （构成了重载而不是重写，remove obj是coll的方法，返回布尔类型的值，remove idx是list的专属方法，返回idx的对象）
改：set(int index, Object ele)，返回原idx的对象，有点类似于替换
查：get(int index)，返回idx对象
插：add(int index, Object ele)，此方法无返回值
长度：size()
遍历：① Iterator迭代器方式
     ② 增强for循环
     ③ 普通的循环

# 3.Set接口

## 3.1 特性

- Set接口中没额外定义新的方法，使用的都是Collection中声明过的方法。因为是无序的，所以没有诸如List接口的get()方法和add()的重载方法。

![](C:\Users\acer-pc\Desktop\办公\自己写的资料\picture\java集合\Set抽象方法.png)

- 存储的数据特点：无序的、不可重复的元素

  1. 无序性：不等于随机性。存储的数据在底层数组中并非照数组索引的顺序添加，而是根据数据的哈希值决定的。
  2. 不可重复性：保证添加的元素照equals()判断时，不能返回true.即：相同的元素只能添加一个。

- 元素添加过程：(以HashSet为例)

  我们向HashSet中添加元素a,首先调用元素a所在类的hashCode()方法，计算元素a的哈希值，此哈希值接着通过某种算法计算出在HashSet底层数组中的存放位置（即为：索引位置，判断数组此位置上是否已经元素：
      如果此位置上没其他元素，则元素a添加成功。 --->情况1
      如果此位置上其他元素b(或以链表形式存在的多个元素，则比较元素a与元素b的hash值：
          如果hash值不相同，则元素a添加成功。--->情况2
          如果hash值相同，进而需要调用元素a所在类的equals()方法：
                 equals()返回true,元素a添加失败
                 equals()返回false,则元素a添加成功。--->情况3

- HashSet底层：数组+链表的结构。（前提：jdk7)

  对于添加成功的情况2和情况3而言：元素a 与已经存在指定索引位置上数据以链表的方式存储。
  jdk 7 :元素a放到数组中，指向原来的元素。
  jdk 8 :原来的元素在数组中，指向元素a
  总结：七上八下



ps：

- 哈希值是通过对象的属性来计算的。相同的属性计算出来的哈希值是一样的。
- Object对象的hashCode()方法（重写前的方法）计算出来的哈希值用于表示对象在内存的存放位置。且计算过程是随机计算的。假如说new了两个对象，且这两个对象都有相同的属性值，但是随机计算出来的哈希值是不同的。
- 属性不同的两个对象，有可能计算出相同的哈希值，因此还需要equals进行判断。
- 对于存放在Set容器中的对象，对应的类一定要重写equals()和hashCode()方法。String类型的变量已经自动重写了两个方法。
- 详情参考[尚硅谷视频](https://www.bilibili.com/video/BV1Kb411W75N?p=537)

# 4. Map

## 4.1 HashMap

HashMap的底层：

- 数组+链表  （jdk7及之前)

- 数组+链表+红黑树 （jdk 8)



## 4.2 常用方法

```java
添加、删除、修改操作：
Object put(Object key,Object value)：将指定key-value添加到(或修改)当前map对象中
void putAll(Map m):将m中的所有key-value对存放到当前map中
Object remove(Object key)：移除指定key的key-value对，并返回value
void clear()：清空当前map中的所有数据
元素查询的操作：
Object get(Object key)：获取指定key对应的value
boolean containsKey(Object key)：是否包含指定的key
boolean containsValue(Object value)：是否包含指定的value
int size()：返回map中key-value对的个数
boolean isEmpty()：判断当前map是否为空
boolean equals(Object obj)：判断当前map和参数对象obj是否相等
元视图操作的方法：
Set keySet()：返回所有key构成的Set集合
Collection values()：返回所有value构成的Collection集合
Set entrySet()：返回所有key-value对构成的Set集合

*总结：常用方法：
* 添加：put(Object key,Object value)
* 删除：remove(Object key)
* 修改：put(Object key,Object value)
* 查询：get(Object key)
* 长度：size()
* 遍历：keySet() / values() / entrySet()
```

```java
   /*
 元素查询的操作：
 Object get(Object key)：获取指定key对应的value
 boolean containsKey(Object key)：是否包含指定的key
 boolean containsValue(Object value)：是否包含指定的value
 int size()：返回map中key-value对的个数
 boolean isEmpty()：判断当前map是否为空
 boolean equals(Object obj)：判断当前map和参数对象obj是否相等
     */
    @Test
    public void test4(){
        Map map = new HashMap();
        map.put("AA",123);
        map.put(45,123);
        map.put("BB",56);
        // Object get(Object key)
        System.out.println(map.get(45));
        //containsKey(Object key)
        boolean isExist = map.containsKey("BB");
        System.out.println(isExist);

        isExist = map.containsValue(123);
        System.out.println(isExist);

        map.clear();

        System.out.println(map.isEmpty());

    }

    /*
     添加、删除、修改操作：
 Object put(Object key,Object value)：将指定key-value添加到(或修改)当前map对象中
 void putAll(Map m):将m中的所有key-value对存放到当前map中
 Object remove(Object key)：移除指定key的key-value对，并返回value
 void clear()：清空当前map中的所有数据
     */
    @Test
    public void test3(){
        Map map = new HashMap();
        //添加
        map.put("AA",123);
        map.put(45,123);
        map.put("BB",56);
        //修改
        map.put("AA",87);

        System.out.println(map);

        Map map1 = new HashMap();
        map1.put("CC",123);
        map1.put("DD",123);

        map.putAll(map1);

        System.out.println(map);

        //remove(Object key)
        Object value = map.remove("CC");
        System.out.println(value);
        System.out.println(map);

        //clear()
        map.clear();//与map = null操作不同
        System.out.println(map.size());
        System.out.println(map);
    }
```



- Map中的key:无序的、不可重复的，使用Set存储所的key  ---> key所在的类要重写equals()和hashCode() （以HashMap为例)
- Map中的value:无序的、可重复的，使用Collection存储所的value --->value所在的类要重写equals()
- 一个键值对：key-value构成了一个Entry对象。
- Map中的entry:无序的、不可重复的，使用Set存储所的entry

```java
    /*
 元视图操作的方法：
 Set keySet()：返回所有key构成的Set集合
 Collection values()：返回所有value构成的Collection集合
 Set entrySet()：返回所有key-value对构成的Set集合

     */


    @Test
    public void test5(){
        Map map = new HashMap();
        map.put("AA",123);
        map.put(45,1234);
        map.put("BB",56);

        //遍历所有的key集：keySet()
        Set set = map.keySet();
            Iterator iterator = set.iterator();
            while(iterator.hasNext()){
                System.out.println(iterator.next());
        }
        System.out.println();
        //遍历所有的value集：values()
        Collection values = map.values();
        for(Object obj : values){
            System.out.println(obj);
        }
        System.out.println();
        //遍历所有的key-value
        //方式一：entrySet()
        Set entrySet = map.entrySet();
        Iterator iterator1 = entrySet.iterator();
        while (iterator1.hasNext()){
            Object obj = iterator1.next();
            //entrySet集合中的元素都是entry
            Map.Entry entry = (Map.Entry) obj;
            System.out.println(entry.getKey() + "---->" + entry.getValue());

        }
        System.out.println();
        //方式二：
        Set keySet = map.keySet();
        Iterator iterator2 = keySet.iterator();
        while(iterator2.hasNext()){
            Object key = iterator2.next();
            Object value = map.get(key);
            System.out.println(key + "=====" + value);

        }

    }
```

### 4.2.1 map.getOrDefault(Object key, V defaultValue)

- getOrDefault() 方法获取指定 key 对应对 value，如果找不到 key ，则返回设置的默认值。

```
import java.util.HashMap;

class Main {
    public static void main(String[] args) {
        // 创建一个 HashMap
        HashMap<Integer, String> sites = new HashMap<>();

        // 往 HashMap 添加一些元素
        sites.put(1, "Google");
        sites.put(2, "Runoob");
        sites.put(3, "Taobao");
        System.out.println("sites HashMap: " + sites);

        // key 的映射存在于 HashMap 中
        // Not Found - 如果 HashMap 中没有该 key，则返回默认值
        String value1 = sites.getOrDefault(1, "Not Found");
        System.out.println("Value for key 1:  " + value1);

        // key 的映射不存在于 HashMap 中
        // Not Found - 如果 HashMap 中没有该 key，则返回默认值
        String value2 = sites.getOrDefault(4, "Not Found");
        System.out.println("Value for key 4: " + value2);
    }
}
```

执行结果

```
Value for key 1:  Google
Value for key 4: Not Found
```

[菜鸟教程](https://www.runoob.com/java/java-hashmap-getordefault.html)



# 5.小技巧

## 5.1 Set<Character>set = map.keySet();

- 利用keySet()返回key

- 这儿加了一个泛型，返回的set就为Character类型了，如果不加泛型，返回的类型就为object类型。比如在力扣383题中：

```
        Set<Character>set = map.keySet();
        for(Character key:set){
            if(map2.containsKey(key)&&(map2.get(key)>=map.get(key))){
                continue;
            }else{
                return false;
            }
        }
```

必须加了泛型，才能进行遍历，否则就会报错

```
Line 16: error: incompatible types: Object cannot be converted to Character
        for(Character key:set){
                          ^
```

