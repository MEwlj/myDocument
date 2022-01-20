# 1. Deque

Deque是一个双端队列（Double Ended Queue），这将意味着它不过是对Queue接口的增强。如果仔细分析Deque接口代码的话，我们会发现它里面主要包含有4个部分的功能定义。1. 双向队列特定方法定义。 2. Queue方法定义。 3. Stack方法定义。4.Collection方法定义。
第3、4部分的方法相当于告诉我们，具体实现Deque的类我们也可以把他们当成Stack和普通的Collection来使用。
————————————————
版权声明：本文为CSDN博主「Lazy别太认真」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/lk941206/article/details/106607785/

### 定义：

虽然Deque继承自Queue，所以有offer等接口，但最好是调用xxxFirst()/xxxLast()以便与Queue的方法区分开。

```
public interface Deque<E> extends Queue<E> {  
//First队首，Last队尾
    //添加
    boolean offerFirst(E e);
    boolean offerLast(E e);
    void addFirst(E e);
    void addLast(E e);
    //取元素并删除
    E pollFirst();
    E pollLast();
    E removeFirst();
    E removeLast();
    //取元素但不删除
    E peekFirst()；
    E peekLast()；
    E getFirst()；
    E getLast()；
}
```

### 构造：

Deque是一个接口，它的实现类有ArrayDeque和LinkedList。

##### LinkedList实现

双端队列：

```
Deque<String> deque = new LinkedList<>();
```

栈：
把Deque作为Stack使用时，注意只调用push()/pop()/peek()方法，不要调用addFirst()/removeFirst()/peekFirst()方法，这样代码更加清晰。

```
Deque<String> stack = new LinkedList<>();

//把元素压栈
stack.push(E)     
//stack.addFirst(e);

//把栈顶的元素“弹出”
E=stack.pop()     
//stack.removeFirst();

//取栈顶元素但不弹出
E=stack.peek()    
stack.peekFirst() 
```

###### ArrayDeque实现

不同于链表List，数组增删元素的操作要考虑数组下标，同时也需要考虑扩展数组空间的问题。

```
Deque<String> deque = new ArrayDeque<>();
```



# 2. Queue

不要把null添加到队列中，否则poll()方法返回null时，很难确定是取到了null元素还是队列为空。
Queue实际上是实现了一个先进先出（FIFO：First In First Out）的有序表。它和List的区别在于，List可以在任意位置添加和删除元素，而Queue只有两个操作：

- 把元素添加到队列末尾；

- 从队列头部取出元素。

定义：

```
public interface Queue<E> extends Collection<E> {  
    boolean offer(E e);  //添加元素到队列中，添加失败false
    add(E e); // 添加元素到队列中，添加失败抛出异常 
    E poll();  //移除队头元素 ，返回false或null
    E remove(); //移除队头元素 ，throw Exception
    E peek();  //获取但不移除队列头的元素 ，返回false或null
    E element(); //获取但不移除队列头的元素 ，throw Exception
} 
```

————————————————
版权声明：本文为CSDN博主「Lazy别太认真」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/lk941206/article/details/106607785/

补充一点：queue.offer()和add()的区别

- Queue 中 add() 和 offer()都是用来向队列添加一个元素。
- 在容量已满的情况下，add() 方法会抛出IllegalStateException异常，offer() 方法只会返回 false 。

offer，add区别：
一些队列有大小限制，因此如果想在一个满的队列中加入一个新项，多出的项就会被拒绝。
这时新的 offer 方法就可以起作用了。它不是对调用 add() 方法抛出一个 unchecked 异常，而只是得到由 offer() 返回的 false。

poll，remove区别：
remove() 和 poll() 方法都是从队列中删除第一个元素。remove() 的行为与 Collection 接口的版本相似，
但是新的 poll() 方法在用空集合调用时不是抛出异常，只是返回 null。因此新的方法更适合容易出现异常条件的情况。

peek，element区别：
element() 和 peek() 用于在队列的头部查询元素。与 remove() 方法类似，在队列为空时， element() 抛出一个异常，而 peek() 返回 null

[csdn](https://blog.csdn.net/qq_31963719/article/details/78171170)
