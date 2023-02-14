# Command

The intent of the command pattern is to turn commands into stand-alone objects so that the object invoking the command does not need to worry about how the command is done. By doing so you can delay, queue, undo commands.

## Structure

```plantuml
@startuml
!theme purplerain from https://raw.githubusercontent.com/LuciferUchiha/georgerowlands.ch/main

Command <-- Invoker  : calls
Command <|-- CommandImpl
Receiver <-- CommandImpl : calls

class Invoker {
    void setCommand(Command c)
    void execute()
}

interface Command{
    void execute()
    void undo()
}

class Receiver{
    action()
}

@enduml
```

## Example

```java
// Client
public class CommandPatternDemo {
   public static void main(String[] args) {
      Stock abcStock = new Stock();

      BuyStock buyStockOrder = new BuyStock(abcStock);
      SellStock sellStockOrder = new SellStock(abcStock);

      Broker broker = new Broker();
      broker.takeOrder(buyStockOrder);
      broker.takeOrder(sellStockOrder);

      broker.placeOrders();
   }
}
// Command Interface
public interface Order {
 void execute(); 
}
// Receiver
public class Stock {
   private String name = "ABC";
   private int quantity = 10;

   public void buy(){
      System.out.println("Stock [ Name: "+name+", 
         Quantity: " + quantity +" ] bought");
   }
   public void sell(){
      System.out.println("Stock [ Name: "+name+", 
         Quantity: " + quantity +" ] sold");
   }
}
// ConcreteCommands
public class BuyStock implements Order {
   private Stock abcStock;

   public BuyStock(Stock abcStock){
      this.abcStock = abcStock;
   }

   public void execute() {
      abcStock.buy();
   }
}
public class SellStock implements Order {
   private Stock abcStock;

   public SellStock(Stock abcStock){
      this.abcStock = abcStock;
   }

   public void execute() {
      abcStock.sell();
   }
}  
// Invoker
public class Broker {
   private List<Order> orderList = new ArrayList<Order>(); 

   public void takeOrder(Order order){
      orderList.add(order);  
   }

   public void placeOrders(){
      for (Order order : orderList) {
         order.execute();
      }
      orderList.clear();
   }
}
```
