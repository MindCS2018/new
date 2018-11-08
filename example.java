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
