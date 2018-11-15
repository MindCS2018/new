//exp.1
import java.util.*;
public class PrintArrayInJava {

   public static void main(String[] args) {
      int[] testArray = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
      String operators = "+-*/";
      System.out.println(operators.length());
      System.out.println(operators.charAt(1));
      
      System.out.println(Arrays.toString(testArray));

   }
}

//exp 2
import java.util.*;
public class StackDemo {

   static void showpush(Stack st, int a) {
      st.push(new Integer(a));
      System.out.println("push(" + a + ")");
      System.out.println("stack: " + st);
   }

   static void showpop(Stack st) {
      System.out.print("pop -> ");
      Integer a = (Integer) st.pop();
      System.out.println(a);
      System.out.println("stack: " + st);
   }

   public static void main(String args[]) {
      //String[] myArray = new String[2];
      //int[] intArray = new int[2];

      // or can be declared as follows
      String[] myArray = {"this", "is", "my", "array"};
      int[] intArray = {1,2,3,4}; 
      
      ArrayList<String> myList = new ArrayList<String>();
      myList.add("Hello");
      myList.add("World");

      ArrayList<Integer> myNum = new ArrayList<Integer>();
      myNum.add(1);
      myNum.add(2);
      System.out.println(Arrays.toString(myArray));
      System.out.println(Arrays.toString(intArray));
      System.out.println("Using for loop");
	  System.out.println("--------------");
	  for (int i = 0; i < myList.size(); i++) {
			System.out.println(myList.get(i));
		}
     

      Stack st = new Stack();
      System.out.println("stack: " + st);
      showpush(st, 42);
      showpush(st, 66);
      showpush(st, 99);
      showpop(st);
      showpop(st);
      showpop(st);
      try {
         showpop(st);
      } catch (EmptyStackException e) {
         System.out.println("empty stack");
      }
   }
}

//exp.3
package com.mkyong.utils.print;

import java.util.Arrays;

public class PrintArray {

    public static void main(String[] args) {

		// array
        String[] arrayStr = new String[]{"Java", "Node", "Python", "Ruby"};
        System.out.println(Arrays.toString(arrayStr));
        // Output : [Java, Node, Python, Ruby]

		int[] arrayInt = {1, 3, 5, 7, 9};
        System.out.println(Arrays.toString(arrayInt));
        // Output : [1, 3, 5, 7, 9]

        // 2d array, need Arrays.deepToString
        String[][] deepArrayStr = new String[][]{{"mkyong1", "mkyong2"}, {"mkyong3", "mkyong4"}};
        System.out.println(Arrays.toString(deepArrayStr));
        // Output : [[Ljava.lang.String;@23fc625e, [Ljava.lang.String;@3f99bd52]

        System.out.println(Arrays.deepToString(deepArrayStr));
        // Output : [[mkyong1, mkyong2], [mkyong3, mkyong4]

        int[][] deepArrayInt = new int[][]{{1, 3, 5, 7, 9}, {2, 4, 6, 8, 10}};
        System.out.println(Arrays.toString(deepArrayInt));
        // Output : [[I@3a71f4dd, [I@7adf9f5f]

        System.out.println(Arrays.deepToString(deepArrayInt));
        // Output : [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]

    }

}

//exp.4

import java.util.ArrayDeque;

public class Program {
    public static void main(String[] args) {

        // Create ArrayDeque with three elements.
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        deque.push(10);
        deque.push(500);
        deque.push(1000);

        // Peek to get the top item, but do not remove it.
        int peekResult = deque.peek();
        System.out.println(peekResult);

        // Call pop on the Deque.
        int popResult = deque.pop();
        System.out.println(popResult);

        // Pop again.
        popResult = deque.pop();
        System.out.println(popResult);
    }
}

//exp.5

import java.util.ArrayDeque;


public class Program {
    public static void main(String[] args) {

        ArrayDeque<String> deque = new ArrayDeque<>();

        // Use add on ArrayDeque.
        deque.add("caterpillar");
        deque.add("dinosaur");
        deque.add("bird");

        // Loop over all the elements in the ArrayDeque.
        for (String element : deque) {
            System.out.println(element);
        }
    }
}

//exp.6 

  
import java.util.*; 
  
public class Test 
{ 
    public static void main(String args[]) 
    { 
        // Creating object of class linked list 
        LinkedList<String> object = new LinkedList<String>(); 
  
        // Adding elements to the linked list 
        object.add("A"); 
        object.add("B"); 
        object.addLast("C"); 
        object.addFirst("D"); 
        object.add(2, "E"); 
        object.add("F"); 
        object.add("G"); 
        System.out.println("Linked list : " + object); 
  
        // Removing elements from the linked list 
        object.remove("B"); 
        object.remove(3); 
        object.removeFirst(); 
        object.removeLast(); 
        System.out.println("Linked list after deletion: " + object); 
  
        // Finding elements in the linked list 
        boolean status = object.contains("E"); 
  
        if(status) 
            System.out.println("List contains the element 'E' "); 
        else
            System.out.println("List doesn't contain the element 'E'"); 
  
        // Number of elements in the linked list 
        int size = object.size(); 
        System.out.println("Size of linked list = " + size); 
  
        // Get and set elements from linked list 
        Object element = object.get(2); 
        System.out.println("Element returned by get() : " + element); 
        object.set(2, "Y"); 
        System.out.println("Linked list after change : " + object); 
    } 
}

//exp.7

public class Add{
  public static int add_int(int x,int y){
    return x+y;
  }
  public static void main(String[] args){
    int z;
    z = add_int(2,4);
    System.out.println("The result is:" +z);
    System.out.println(z);
  }
}

//exp.8

import java.util.*;
public class Add{
  public static void main(String[] args){
      
    //String[] myArray = new String[2];
    //int[] intArray = new int[2];

    // or can be declared as follows
    String[] myArray = {"this", "is", "my", "array"};
    int[] intArray = {1,2,3,4};


   ArrayList<String> myList = new ArrayList<String>();
   myList.add("Hello");
   myList.add("World");

   ArrayList<Integer> myNum = new ArrayList<Integer>();
   myNum.add(1);
   myNum.add(2);


   Stack myStack = new Stack();
   // add any type of elements (String, int, etc..)
   myStack.push("Hello");
   myStack.push(1);
 
   
   Queue<String> myQueue = new LinkedList<String>();
   Queue<Integer> myNumbers = new LinkedList<Integer>();
   myQueue.add("Hello");
   myQueue.add("World");
   myNumbers.add(1);
   myNumbers.add(2);
  }
}


//exp.9

import java.util.*;
public class FindLongestPalindrom{
  public static boolean isPalindrom(String s){
      for (int i=0;i<s.length()-1;i++){
          if (s.charAt(i)!=s.charAt(s.length()-1-i)){
              return false;
          }
      }
      return true;
  }
  public static int longestPalindrom(String s){
      int maxPalindromLen=0;
      String longestPalindrom=null;
      int length=s.length();
      for (int i=0;i<length;i++){
          for (int j=i+1;j<length;j++){
              int len=j-i;
              String curr=s.substring(i,j+1);
              if (isPalindrom(curr)){
                  if (len>maxPalindromLen){
                      longestPalindrom=curr;
                      maxPalindromLen=len;
                  }
              }
          }
      }
      return maxPalindromLen;
  }
  public static void main(String[] args){
      
    String s = "bcba";
    int maxLen=longestPalindrom(s);
    System.out.println(maxLen);


  }
}

//exp.10

import java.util.*;
public class Solution{
  public static boolean wordBreak(String s, Set<String> dict){
      return wordBreakHelper(s,dict,0);
  }
  public static boolean wordBreakHelper(String s, Set<String> dict, int start){
      if (start==s.length()){
          return true;
      }
      for (String a: dict){
          int len=a.length();
          int end=start+len;
          if (end>s.length())
             continue;
          if (s.substring(start,start+len).equals(a))
             if (wordBreakHelper(s,dict,start+len))
                return true;
      }
      return false;
  }
  public static void main(String[] args){
      
    String s = "leetcode";
    Set dict = new HashSet();
    dict.add("leet");
    dict.add("code");

    //Set<String> dict=new HashMap<String>();
    //dict.put("leet","code");
    boolean o=wordBreak(s,dict);
    System.out.println(o);


  }
}


//2018.11.13
import java.util.*;
import java.util.*; 
public class Solution{
  public static int ladderLength(String start, String end, HashSet<String> dict){
      if (dict.size()==0)
         return 0;
      dict.add(end);
      LinkedList<String> wordQueue= new LinkedList<String>();
      LinkedList<Integer> wordDistance=new LinkedList<Integer>();
      wordQueue.add(start);
      wordDistance.add(1);
      int result=Integer.MAX_VALUE; 
      while (!wordQueue.isEmpty()){
          String currWord=wordQueue.pop();
          Integer currDistance=wordDistance.pop();
          System.out.println(currWord);
          if (currWord.equals(end)){
              result=currDistance;
          }
          for (int i=0;i<currWord.length();i++){
              char[] currCharArr=currWord.toCharArray();
              for (char c='a';c<'z';c++){
                  currCharArr[i]=c;
                  String newWord=new String(currCharArr);
                  if (dict.contains(newWord)){
                      wordQueue.add(newWord);
                      wordDistance.add(currDistance+1);
                      dict.remove(newWord);
                  }
              }
          }
      }
      if (result<Integer.MAX_VALUE)
          return result;
      else
          return 0;
          
  }  

  public static void main(String[] args){
      
    String start = "hit";
    String end="cog";
    HashSet<String> dict = new HashSet<String>();
    dict.add("hot");
    dict.add("dot");
    dict.add("dog");
    dict.add("lot");
    dict.add("log");
    
    System.out.println(dict);

    //Set<String> dict=new HashMap<String>();
    //dict.put("leet","code");
    int o=ladderLength(start,end,dict);
    System.out.println(o);


  }
}


import java.util.*;
import java.util.*; 
public class Solution{
  public static double findMedianTwoSortedArray(int A[], int B[]){
  
     int m=A.length;
     int n=B.length;
     if ((m+n)%2!=0)
      return (double) findKth(A,B,(m+n)/2,0,m-1,0,n-1);
     else{
       double var1=findKth(A,B,(m+n)/2,0,m-1,0,n-1);
       double var2=findKth(A,B,(m+n)/2-1,0,m-1,0,n-1);
      System.out.println(var2);
      return (var1+var2)*0.5;
     }
  } 
  public static int findKth(int A[], int B[], int k, int aStart,int aEnd, int bStart,int bEnd){
      int alen=aEnd-aStart+1;
      int blen=bEnd-bStart+1;
      if (alen==0)
         return B[bStart+k];
      if (blen==0)
         return A[aStart+k];
      if (k==0)
         return A[aStart]<B[bStart]?A[aStart]:B[bStart];
      int aMid=alen*k/(alen+blen);
      int bMid=k-aMid-1;
      
      aMid=aMid+aStart;
      bMid=bMid+bStart;
      
      if (A[aMid]>B[bMid]){
          k=k-(bMid-bStart+1);
          aEnd=aMid;
          bStart=bMid+1;
      } else {
          k=k-(aMid-aStart+1);
          bEnd=bMid;
          aStart=aMid+1;
          
      }
      return findKth(A,B,k,aStart,aEnd,bStart,bEnd);
         
  }

  public static void main(String[] args){
    int[] A = {1,2,3,4,5,6};//{1,3};
    int[] B = {2,3,4,5};//{2};
    double k=findMedianTwoSortedArray(A, B);
    
    System.out.println(Arrays.toString(A));
    System.out.println(Arrays.toString(B));
    System.out.println(k);

    //Set<String> dict=new HashMap<String>();
    //dict.put("leet","code");
    //int o=ladderLength(start,end,dict);
    //System.out.println(o);


  }
}
