import java.util.Arrays;
public class PrintArrayInJava {
   private static void printArray(int[] anArray) {
      for (int i = 0; i < anArray.length; i++) {
         if (i > 0) {
            System.out.print(", ");
         }
         System.out.print(anArray[i]);
      }
   }
   public static void rotate(int[] nums, int k) {
       int[] result= new int[nums.length];
       for (int i=0;i<k;i++){
           result[i]=nums[nums.length-k+i];
       }
       
       int j=0;
       for (int i=k;i<nums.length;i++){
           result[i]=nums[j];
           j=j+1;
       }
       System.out.println(Arrays.toString(nums));
       System.out.println(Arrays.toString(result));
   }
   public static void main(String[] args) {
      int[] testArray = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
      printArray(testArray);
      int[] intArray = { 7, 9, 5, 1, 3 };
      System.out.println("\n");
      System.out.println(Arrays.toString(intArray));
      rotate(testArray,3);
   }
}
