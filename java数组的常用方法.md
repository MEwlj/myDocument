# 1. Arrays.toString([])

用于打印数组元素的值

例如

```
System.out.println(Arrays.toString(res));
```



# 2.Arrays.sort([])

1. 对数组元素进行排序
2. 无返回值。



# 3.Arrays.copyOfRange(T[ ] original,int from,int to)

1. 将一个原始的数组original，从下标from开始复制，复制到上标to，生成一个新的数组。
2. 开区间。

该命令可用户返回值内，例如力扣的第350题。

```
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0;int j = 0;
        int length1 = nums1.length;
        int length2 = nums2.length;
        int res[] = new int[Math.min(nums1.length,nums2.length)];
        int index = 0;
        for(;(i<length1) && (j<length2);){
            if(nums1[i]>nums2[j]){
                j++;
            }else if(nums1[i]<nums2[j]){
                i++;
            }else{
                res[index] = nums1[i];
                index++;
                i++;
                j++;
            }
        }
        return Arrays.copyOfRange(res,0,index);
        
    }
}
```

