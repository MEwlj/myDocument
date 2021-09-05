内容选取leetcode上面的代码随想录。

## 39 组合总和

代码

```java
class Solution {
    List<List<Integer>> lists = new ArrayList<>();
    int sum;
    List<Integer> tmp = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        fun(candidates,target,0);
        return lists;
    }

    public void fun(int[] candidates, int target, int idx){
        if(sum == target){
            //必须要new ArrayList
            lists.add(new ArrayList<>(tmp));
            return;
        }

        for(int i = idx;i < candidates.length;i++){
            if(sum + candidates[i] > target)
                break;
            sum += candidates[i];
            tmp.add(candidates[i]);
            //回溯时，startindex变为i
            fun(candidates,target,i);
            tmp.remove(tmp.size() - 1);
            sum -= candidates[i];

        }
    }
}
```

其中

```
        if(sum == target){
            //必须要new ArrayList
            lists.add(new ArrayList<>(tmp));
            return;
        }
```

1. 这儿可能要说一说，因为lists和tmp都是List类型的，都要先转化为ArrayList，而转化的方法如下。

```
lists.add(new ArrayList<>(tmp));
```

2. 回溯时，其startindex要变为外层for循环的i值，这样是为了避免重复取值。
3. 一开始要进行一个排序，只有排序过后，下面的代码才能起到优化的作用，一旦>target，后面的值就不再考虑。

```
if(sum + candidates[i] > target)
	break;
```

4. 对ArrayList取最后的值，可以是tmp.size() - 1，避免了采用Deque的addLast，对比第40题。

```
tmp.remove(tmp.size() - 1);
```



总结一下，该算法有点类似dfs，从某个值开始一直取相同的值，取到最深的深度，然后逐渐回溯。其树状图如下

![39.组合总和](https://pic.leetcode-cn.com/1625367598-JttWEJ-file_1625367598565)



## 40 组合总和II

代码

```java
class Solution {
    List<List<Integer>> lists = new ArrayList<>();
    Deque<Integer> deque = new ArrayDeque<>();
    

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        boolean[] used = new boolean[candidates.length];
        fun(candidates,target,0,used,0);
        return lists;
    }

    public void fun(int[] candidates,int target,int index,boolean[] used,int sum){
        if(sum == target){
            lists.add(new ArrayList<>(deque));
            return;
        }

        for(int i = index;(i < candidates.length)&&(sum < target);i++){
            //i > 0此条件很重要，解决了candidates[i - 1]的报错
            if((i > 0)&&(candidates[i] == candidates[i - 1])&&(!used[i - 1]))
                continue;
            //以下步骤是回溯的核心，只需记住，当给used和sum进行赋值后，再回溯后，将
            //used和sum进行复原，不要去管回溯过程中具体的过程。
            used[i] = true;
            sum += candidates[i];
            deque.addLast(candidates[i]);
            fun(candidates,target,i + 1,used,sum);
            int tmp = deque.removeLast();
            used[i] = false;
            sum -= tmp;
        }
    }
}
```

1. ```
           if(sum == target){
               lists.add(new ArrayList<>(deque));
               return;
           }
   ```

   这儿采用了双向队列Deque，有addLast和removeLast两个方法，一个是在队尾添加元素，另一个是移除队尾元素。其中**new ArrayList<>(deque)**把队列转换成了ArrayList。

2. 因为candidates中的元素不能重复取值，故采用一个used的boolean数组来确定相同的元素是在树层还是在树枝

![40.组合总和II](https://pic.leetcode-cn.com/1625367960-UqkYXa-file_1625367957583)

3. ```
   fun(candidates,target,i + 1,used,sum);
   ```

   递归的时候与39题不同，始终找下一个值，当然前提是要排序后。

4. used[i - 1] == true，说明同一树支candidates[i - 1]使用过
   used[i - 1] == false，说明同一树层candidates[i - 1]使用过

ps：自己其实也提交过代码，极其复杂，而且运行时间超时，完全是在第77题基础上改的，本来用于39题，最后发现自己的代码可能只适合40题。

```java
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        int n = candidates.length;
        Deque<Integer> list = new ArrayDeque<>();
        List<List<Integer>> ls = new ArrayList<>();
        for(int i = 1;i<=n;i++){
            fun(candidates,i,target,0,list,ls);
        }
        int tp = 0;
        for(int i = 0;i < ls.size() - 1 - tp;i++)
            for(int j = i + 1;j < ls.size();j++){
                if(findEqual(ls.get(i),ls.get(j))){
                	int count= 0;
                    for(int tmp = 0;tmp < ls.get(i).size();tmp++){
                        if(ls.get(i).get(tmp) > ls.get(j).get(tmp)){
                            ls.remove(i);
                            j--;
                            tp++;
                            break;
                        }else if(ls.get(i).get(tmp) < ls.get(j).get(tmp)){
                        	ls.remove(j);
                        	j--;
                        	tp++;
                        	break;
                        }else{
                        	count++;
                        }
                        if(count == ls.get(i).size()) {
                        	ls.remove(j);
                        	j--;
                        	tp++;
                        	break;
                        }
                    }
                }
            }

        return ls;
    }


	public void fun(int[] candidates,int num, int target ,int begin, Deque<Integer> list, List<List<Integer>> ls){
        if((list.size()==num)&&(sum(list) == target)){
            ls.add(new ArrayList<>(list));
            return;
        }else if(list.size()==num)
            return;

        for(int i = begin;i < candidates.length;i++){
            list.addLast(candidates[i]);
            fun(candidates,num,target,i+1,list,ls);
            list.removeLast();
        }
            
    }

    public int sum(Deque<Integer> list){
        int sum = 0;
        for(Integer tmp:list){
            sum += tmp;
        }
        return sum;
    }

    public boolean findEqual(List<Integer> a,List<Integer> b){
        int[] num = new int[100];
        for(Integer i:a){
            num[i]++;
        }
        for(Integer j:b){
            num[j]--;
        }
        for(int tmp =0;tmp < 100;tmp++)
            if(num[tmp] != 0)
                return false;
        return true;
    }
}
```

此代码是先列举数组中所有组合，判断所有组合的sum是否满足target。

难点在于无法去重。比如说[2,1,3]和[2,3,1]都被保留了，可如果采用HashSet，那么HashSet<HashSet<Integer>>类型的值，无法保留类似[2,1,1]的组合。

- 有一点需要注意下

```java
        for(int i = 0;i < ls.size() - 1 - tp;i++)
            for(int j = i + 1;j < ls.size();j++){
                if(findEqual(ls.get(i),ls.get(j))){
                	int count= 0;
                    for(int tmp = 0;tmp < ls.get(i).size();tmp++){
                        if(ls.get(i).get(tmp) > ls.get(j).get(tmp)){
                            ls.remove(i);
                            j--;
                            tp++;
                            break;
                        }else if(ls.get(i).get(tmp) < ls.get(j).get(tmp)){
                        	ls.remove(j);
                        	j--;
                        	tp++;
                        	break;
                        }else{
                        	count++;
                        }
                        if(count == ls.get(i).size()) {
                        	ls.remove(j);
                        	j--;
                        	tp++;
                        	break;
                        }
                    }
                }
            }

        return ls;
```

这是在去重，假如两个ArrayList中的集合元素相同，但顺序不同，保留第一个不同元素，且元素值较小的那个集合（好像没什么意义，因为可能最后还是乱序的）。但是每移除一个元素，集合的大小会减去1，因此for循环的判断条件也会改变。用**j--**来抵消变化。

## 77 组合

代码

```
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> list  = new ArrayList<List<Integer>>();
        //这儿Deque是双向队列
        Deque<Integer> tmp = new ArrayDeque<>();
        if(n < k || k <0)
            return null;
        fun(n,k,1,tmp,list);
        return list;
    }

    public void fun(int n,int k,int begin,Deque<Integer> path, List<List<Integer>> res){
        if(path.size() == k){
            //？？？
            res.add(new ArrayList<>(path));
            return;
        }

        for(int i = begin;i <= n;i++){
            path.addLast(i);
            fun(n,k,i + 1,path,res);
            path.removeLast();
        }
    }

}
```

![77.组合.png](https://pic.leetcode-cn.com/1603889327-igUOld-77.%E7%BB%84%E5%90%88.png)

## 216 组合总和III

```java
class Solution {
        List<List<Integer>> lists = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        int sum = 0;

    public List<List<Integer>> combinationSum3(int k, int n) {
        int[] arr = new int[10];
        for(int i = 1;i < arr.length;i++){
            arr[i] = i;
        }
        fun(arr,n,k,1);
        return lists;

    }

    public void fun(int[] arr, int n, int k, int startIndex){
        if((sum == n)&&tmp.size()==k){
            lists.add(new ArrayList<>(tmp));
            return;
        }else if(tmp.size()==k){
            return;
        }

        for(int i = startIndex;i <=9;i++){
            sum += arr[i];
            tmp.add(arr[i]);
            fun(arr,n,k,i+1);
            sum -= arr[i];
            tmp.remove(tmp.size() - 1);
        }
    }
}
```

和第77题一样，只不过加了一个**sum==n**的判断条件



## 377 组合总和IV

代码演示

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        //有点没搞懂为什么组合可以用这种动态规划来做，但是纸上写了一遍又确实没错，还是有点似懂非懂。
        int[] arr = new int[target + 1];
        arr[0] = 1;
        for(int i = 1;i <= target;i++){
            for(int j : nums){
                if(i >= j){
                    arr[i] += arr[i - j];
                }
            }
        }
        return arr[target];
    }


}
```

这次是用动态规划来做的，记住外层for循环是target容量，内层for循环是nums的数值，只有这样，才能出现重复的组合，此题与518零钱兑换II非常相似，但是518题不允许有重复组合，只需要更改内外for循环即可。

ps：此题似乎可以根据第39题组合总和来进行修改

```java

class Solution {

    int sum;
    List<Integer> tmp = new ArrayList<>();
    int count;

    public int combinationSum4(int[] nums, int target) {
        Arrays.sort(nums);
        fun(nums,target,0);
        return count;
    }

    public void fun(int[] nums, int target, int idx){
        if(sum == target){
           
 
            count++;
            return;
        }

        for(int i = idx;i < nums.length;i++){
            if(sum + nums[i] > target)
                break;
            sum += nums[i];
            tmp.add(nums[i]);
            fun(nums,target,idx);
            tmp.remove(tmp.size() - 1);
            sum -= nums[i];

        }
    }
}

```

仅仅把递归中的startIndex修改为idx即可，即每次递归都从最开始的startIndex开始递归，但是这样会超出时间限制和内存限制。

## 518 零钱兑换II

代码演示

```java
class Solution {
    public int change(int amount, int[] coins) {
        int arr[] = new int[amount + 1];
        arr[0] = 1;
        Arrays.sort(coins);
        for(int coin: coins){
            for(int tmp = coin;tmp <= amount;tmp++){
                if(tmp >= coin){
                    arr[tmp] += arr[tmp - coin];
                }
            }
        }
        return arr[amount];
    }
}
```

与377题极为类似，但是把内外两层for循环颠倒，就避免了重复的组合。



## 322 零钱兑换

代码演示

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] arr = new int[amount + 1];
        // arr[0] = 1;  这句代码没用
        Arrays.sort(coins);
        if(amount == 0)
            return 0;
        
        for(int i = 1;i<= amount;i++){
            int min = Integer.MAX_VALUE;
            for(int j : coins){
                if(i < j){
                    break;//这句代码对算法进行了优化
                }else if(i == j){
                    arr[i] = 1;
                }else if((i > j)&&(arr[i - j] != 0)&&(arr[i - j] + 1 < min)){
                    arr[i] = 1 + arr[i - j];
                    min = arr[i];
                }
            }
        }
        if(arr[amount] != 0)
            return arr[amount];
        return -1;
    }
}
```

采用dp算法，中间的if-else显得冗余了。关键点是要设置

```
int min = Integer.MAX_VALUE;
```

在dp中不断更新min值。这种写法克服了arr[i] = Math.min(arr[i - j] + 1,arr[i])中，首次给arr[i]赋值时，arr[i]值为0的缺陷。



## 17 电话号码的数字组合

题目：

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/original_images/17_telephone_keypad.png)

代码演示：

```java
class Solution {
    List<String> lists = new ArrayList<>();
    Map<Character,String> map;
    public List<String> letterCombinations(String digits) {
        if((digits.isEmpty())||(digits.trim().isEmpty())){
            return lists;
        }
        map = new HashMap<>();
        
        map.put('2',"abc");
        map.put('3',"def");
        map.put('4',"ghi");
        map.put('5',"jkl");
        map.put('6',"mno");
        map.put('7',"pqrs");
        map.put('8',"tuv");
        map.put('9',"wxyz");
        fun("",0,digits);
        return lists;
    }

    public void fun(String str,int index,String digits){
        if(str.length() == digits.length()){
            lists.add(str);
            return;
        }
        Character c = digits.charAt(index);
        String tmp = map.get(c);
        for(int i = 0;i < tmp.length();i++){
           fun(str + tmp.charAt(i),index + 1,digits);

        }
    }
}
```

与之前的题不同的是，该题考查的是不同集合之间，从每个集合各自抽一个数来组成新集合，之前的题都是从一个大集合中找子集合。



## 454 [四数相加 II]

题目：给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。

为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -2^28 到 2^28 - 1 之间，最终结果不会超过 2^31 - 1 。

示例代码：

```java
class Solution {
    int count = 0;

    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer,Integer> map = new HashMap<>();
        for(int i = 0;i < nums1.length;i++){
            for(int j = 0;j < nums2.length;j++){
                int tmp = nums1[i] + nums2[j];
                if(map.containsKey(tmp)){
                    map.put(tmp,map.get(tmp) + 1);
                }else{
                    map.put(tmp,1);
                }
            }
        }

        for(int i = 0;i < nums3.length;i++){
            for(int j = 0;j < nums4.length;j++){
                int tmp = nums3[i] + nums4[j];
                if(map.containsKey(-tmp)){
                    count += map.get(-tmp);
                }
            }
        }

        return count;
    }

    
}
```

该题很可能可以用第17题的思路来做，有4个长度相同的组合，从买个组合各自取一个数作为新的组合，若新组合之和为零，则计数一次。但是问题就在于不知道如何把这4个数组放到同一个集合内。



根据题解：

1. 一采用分为两组，HashMap 存一组，另一组和 HashMap 进行比对。

2. 这样的话情况就可以分为三种：

- HashMap 存一个数组，如 A。然后计算三个数组之和，如 BCD。时间复杂度为：O(n)+O(n^3)，得到 O(n^3).
- HashMap 存三个数组之和，如 ABC。然后计算一个数组，如 D。时间复杂度为：O(n^3)+O(n)，得到 O(n^3).
- HashMap存两个数组之和，如AB。然后计算两个数组之和，如 CD。时间复杂度为：O(n^2)+O(n^2)，得到 O(n^2).

3. 根据第二点我们可以得出要存两个数组算两个数组。

4. 我们以存 AB 两数组之和为例。首先求出 A 和 B 任意两数之和 sumAB，以 sumAB 为 key，sumAB 出现的次数为 value，存入 hashmap 中。
   然后计算 C 和 D 中任意两数之和的相反数 sumCD，在 hashmap 中查找是否存在 key 为 sumCD。
   算法时间复杂度为 O(n2)。

## 15 三数之和

题目：给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

思路：此题可以照第40题的思路来做，仅仅需要改一个条件就行了，但是在测试用例时，加入了一个非常长的测试用例，导致用第40题的思路会超时，因此得换成用hashmap的办法来做，空间换时间。
